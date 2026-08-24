//! PutItem / GetItem / DeleteItem / UpdateItem against a BigTable data table.

use std::collections::BTreeMap;

use extenddb_core::expression::{
    Expr, ExpressionMaps, UpdateAction, apply_update, evaluate_condition,
};
use extenddb_core::types::{Item, TableKeyInfo};
use extenddb_storage::error::StorageError;
use googleapis_tonic_google_bigtable_v2::google::bigtable::v2::mutation::SetCell;
use googleapis_tonic_google_bigtable_v2::google::bigtable::v2::row_filter::{
    Chain, Condition, Filter, Interleave,
};
use googleapis_tonic_google_bigtable_v2::google::bigtable::v2::{
    CheckAndMutateRowRequest, MutateRowRequest, Mutation, ReadRowsRequest, RowFilter, RowSet,
    TimestampRange, mutation,
};

use crate::data::client::BigtableClient;
use crate::data::encoding::{cell, row_key};

/// Family for DDB attribute cells.
pub const FAMILY_DATA: &str = "d";
/// Family for metadata cells (OCC row version, 2PC intents).
pub const FAMILY_META: &str = "m";
/// Qualifier for OCC row version cell in family 'm'.
pub const QUALIFIER_VERSION: &[u8] = b"v";
/// Qualifier prefix for 2PC intent cells in family 'm'.
pub const QUALIFIER_INTENT_PREFIX: &[u8] = b"intent:";

/// Decodes an OCC row version from cell bytes.
pub fn decode_version(bytes: &[u8]) -> Option<u64> {
    if bytes.len() == 8 {
        Some(u64::from_be_bytes(bytes.try_into().unwrap()))
    } else {
        std::str::from_utf8(bytes).ok()?.parse::<u64>().ok()
    }
}

/// Filter matching any active intent in family 'm'.
fn build_active_intent_filter(cutoff_micros: i64) -> RowFilter {
    RowFilter {
        filter: Some(Filter::Chain(Chain {
            filters: vec![
                RowFilter {
                    filter: Some(Filter::FamilyNameRegexFilter(format!("^{FAMILY_META}$"))),
                },
                RowFilter {
                    filter: Some(Filter::ColumnQualifierRegexFilter(b"^intent:.*".to_vec())),
                },
                RowFilter {
                    filter: Some(Filter::CellsPerColumnLimitFilter(1)),
                },
                RowFilter {
                    filter: Some(Filter::TimestampRangeFilter(TimestampRange {
                        start_timestamp_micros: cutoff_micros.max(0),
                        end_timestamp_micros: 0,
                    })),
                },
            ],
        })),
    }
}

/// Filter matching row version m:v == expected_version.
fn build_version_match_filter(expected_version: u64) -> RowFilter {
    let val_regex = format!("^{}$", expected_version).into_bytes();
    RowFilter {
        filter: Some(Filter::Chain(Chain {
            filters: vec![
                RowFilter {
                    filter: Some(Filter::FamilyNameRegexFilter(format!("^{FAMILY_META}$"))),
                },
                RowFilter {
                    filter: Some(Filter::ColumnQualifierRegexFilter(b"^v$".to_vec())),
                },
                RowFilter {
                    filter: Some(Filter::CellsPerColumnLimitFilter(1)),
                },
                RowFilter {
                    filter: Some(Filter::ValueRegexFilter(val_regex)),
                },
            ],
        })),
    }
}

/// Filter matching if row version m:v != expected_version (or m:v is missing).
fn build_version_mismatch_filter(expected_version: u64) -> RowFilter {
    RowFilter {
        filter: Some(Filter::Condition(Box::new(Condition {
            predicate_filter: Some(Box::new(build_version_match_filter(expected_version))),
            true_filter: Some(Box::new(RowFilter {
                filter: Some(Filter::BlockAllFilter(true)),
            })),
            false_filter: Some(Box::new(RowFilter {
                filter: Some(Filter::PassAllFilter(true)),
            })),
        }))),
    }
}

/// De Morgan conflict filter for existing-row OCC update/delete:
/// Matches if (active intent exists in m) OR (m:v != expected_version).
/// When this filter matches (conflict), mutations are skipped.
/// When this filter does NOT match (no conflict), false_mutations are applied.
fn build_occ_conflict_filter(cutoff_micros: i64, expected_version: u64) -> RowFilter {
    RowFilter {
        filter: Some(Filter::Interleave(Interleave {
            filters: vec![
                build_active_intent_filter(cutoff_micros),
                build_version_mismatch_filter(expected_version),
            ],
        })),
    }
}

/// Filter matching if active intent exists OR m:v exists OR data family d exists.
fn build_row_presence_conflict_filter(cutoff_micros: i64) -> RowFilter {
    RowFilter {
        filter: Some(Filter::Interleave(Interleave {
            filters: vec![
                build_active_intent_filter(cutoff_micros),
                RowFilter {
                    filter: Some(Filter::Chain(Chain {
                        filters: vec![
                            RowFilter {
                                filter: Some(Filter::FamilyNameRegexFilter(format!(
                                    "^{FAMILY_META}$"
                                ))),
                            },
                            RowFilter {
                                filter: Some(Filter::ColumnQualifierRegexFilter(b"^v$".to_vec())),
                            },
                            RowFilter {
                                filter: Some(Filter::CellsPerColumnLimitFilter(1)),
                            },
                        ],
                    })),
                },
                RowFilter {
                    filter: Some(Filter::Chain(Chain {
                        filters: vec![
                            RowFilter {
                                filter: Some(Filter::FamilyNameRegexFilter(format!(
                                    "^{FAMILY_DATA}$"
                                ))),
                            },
                            RowFilter {
                                filter: Some(Filter::CellsPerColumnLimitFilter(1)),
                            },
                        ],
                    })),
                },
            ],
        })),
    }
}

/// Holds the table-specific context for a sequence of data operations.
pub struct ItemOps<'a> {
    client: &'a BigtableClient,
    full_table_name: String,
    intent_timeout_secs: u64,
}

impl<'a> ItemOps<'a> {
    pub fn new(
        client: &'a BigtableClient,
        data_table_short: &str,
        intent_timeout_secs: u64,
    ) -> Self {
        Self {
            client,
            full_table_name: client.full_table_name(data_table_short),
            intent_timeout_secs,
        }
    }

    /// Read the entire row into an Item map along with its captured OCC row version.
    /// Returns Ok((None, None)) if absent.
    pub async fn get_with_version(
        &self,
        key_info: &TableKeyInfo,
        key: &Item,
    ) -> Result<(Option<Item>, Option<u64>), StorageError> {
        let row_key = row_key::encode_key(key, &key_info.key_schema)?;
        let mut data = self.client.data();
        let req = ReadRowsRequest {
            table_name: self.full_table_name.clone(),
            app_profile_id: self.client.app_profile_id.clone().unwrap_or_default(),
            rows_limit: 1,
            rows: Some(RowSet {
                row_keys: vec![row_key],
                row_ranges: vec![],
            }),
            filter: Some(RowFilter {
                filter: Some(Filter::CellsPerColumnLimitFilter(1)),
            }),
            ..ReadRowsRequest::default()
        };
        let resp = data
            .read_rows(req)
            .await
            .map_err(|e| StorageError::Internal(format!("ReadRows: {e}")))?;
        let mut iter = resp.into_iter();
        let Some((_, cells)) = iter.next() else {
            return Ok((None, None));
        };
        let mut item: Item = BTreeMap::new();
        let mut version: Option<u64> = None;
        for c in cells {
            if c.family_name == FAMILY_DATA {
                let attr_name = String::from_utf8(c.qualifier)
                    .map_err(|e| StorageError::Internal(format!("decode qualifier: {e}")))?;
                let value = cell::decode(&c.value)?;
                item.insert(attr_name, value);
            } else if c.family_name == FAMILY_META && c.qualifier.as_slice() == QUALIFIER_VERSION {
                version = decode_version(&c.value);
            }
        }
        if item.is_empty() {
            return Ok((None, version));
        }
        Ok((Some(item), version))
    }

    /// Read the entire row into an Item map. Returns Ok(None) if absent.
    pub async fn get(
        &self,
        key_info: &TableKeyInfo,
        key: &Item,
    ) -> Result<Option<Item>, StorageError> {
        let (item, _) = self.get_with_version(key_info, key).await?;
        Ok(item)
    }

    /// Read multiple rows into Item maps. Returns items in the same order as keys.
    /// Absent rows result in None.
    pub async fn batch_get(
        &self,
        key_info: &TableKeyInfo,
        keys: &[Item],
    ) -> Result<Vec<Option<Item>>, StorageError> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut row_keys = Vec::with_capacity(keys.len());
        for k in keys {
            row_keys.push(row_key::encode_key(k, &key_info.key_schema)?);
        }
        let mut data = self.client.data();
        let req = ReadRowsRequest {
            table_name: self.full_table_name.clone(),
            app_profile_id: self.client.app_profile_id.clone().unwrap_or_default(),
            rows_limit: keys.len() as i64,
            rows: Some(RowSet {
                row_keys,
                row_ranges: vec![],
            }),
            filter: Some(RowFilter {
                filter: Some(Filter::CellsPerColumnLimitFilter(1)),
            }),
            ..ReadRowsRequest::default()
        };
        let resp = data
            .read_rows(req)
            .await
            .map_err(|e| StorageError::Internal(format!("ReadRows batch: {e}")))?;

        let mut row_map = BTreeMap::new();
        for (rkey, cells) in resp {
            let mut item: Item = BTreeMap::new();
            for c in cells {
                if c.family_name == FAMILY_DATA {
                    let attr_name = String::from_utf8(c.qualifier)
                        .map_err(|e| StorageError::Internal(format!("decode qualifier: {e}")))?;
                    let value = cell::decode(&c.value)?;
                    item.insert(attr_name, value);
                }
            }
            if !item.is_empty() {
                row_map.insert(rkey, item);
            }
        }

        let mut out = Vec::with_capacity(keys.len());
        for k in keys {
            let rkey = row_key::encode_key(k, &key_info.key_schema)?;
            if let Some(item) = row_map.get(&rkey) {
                out.push(Some(item.clone()));
            } else {
                out.push(None);
            }
        }

        Ok(out)
    }

    /// Read multiple rows into Item maps along with any active intent txn_id in column family `m`.
    /// Returns a vector of tuples: (Option<Item>, Option<String>), where the first element is the
    /// data item decoded from family `d` (if present), and the second element is the active transaction ID
    /// if an unexpired intent marker was found in family `m`.
    pub async fn batch_get_with_intent_check(
        &self,
        key_info: &TableKeyInfo,
        keys: &[Item],
    ) -> Result<Vec<(Option<Item>, Option<String>)>, StorageError> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut row_keys = Vec::with_capacity(keys.len());
        for k in keys {
            row_keys.push(row_key::encode_key(k, &key_info.key_schema)?);
        }
        let mut data = self.client.data();
        let req = ReadRowsRequest {
            table_name: self.full_table_name.clone(),
            app_profile_id: self.client.app_profile_id.clone().unwrap_or_default(),
            rows_limit: keys.len() as i64,
            rows: Some(RowSet {
                row_keys,
                row_ranges: vec![],
            }),
            filter: Some(RowFilter {
                filter: Some(Filter::CellsPerColumnLimitFilter(1)),
            }),
            ..ReadRowsRequest::default()
        };
        let resp = data.read_rows(req).await.map_err(|e| {
            StorageError::Internal(format!("ReadRows batch_get_with_intent_check: {e}"))
        })?;

        let now_micros = (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1000) as i64;
        let cutoff_micros = now_micros - (self.intent_timeout_secs * 1_000_000) as i64;

        let mut row_map: BTreeMap<Vec<u8>, (Option<Item>, Option<String>)> = BTreeMap::new();

        for (rkey, cells) in resp {
            let mut item: Item = BTreeMap::new();
            let mut active_txn_id: Option<String> = None;
            for c in cells {
                if c.family_name == FAMILY_DATA {
                    let attr_name = String::from_utf8(c.qualifier)
                        .map_err(|e| StorageError::Internal(format!("decode qualifier: {e}")))?;
                    let value = cell::decode(&c.value)?;
                    item.insert(attr_name, value);
                } else if c.family_name == FAMILY_META
                    && c.qualifier.starts_with(QUALIFIER_INTENT_PREFIX)
                    && c.timestamp_micros >= cutoff_micros
                    && let Ok(qual_str) = std::str::from_utf8(&c.qualifier)
                    && let Some(tid) = qual_str.strip_prefix("intent:")
                {
                    active_txn_id = Some(tid.to_string());
                }
            }
            let item_opt = if item.is_empty() { None } else { Some(item) };
            row_map.insert(rkey, (item_opt, active_txn_id));
        }

        let mut out = Vec::with_capacity(keys.len());
        for k in keys {
            let rkey = row_key::encode_key(k, &key_info.key_schema)?;
            if let Some(entry) = row_map.get(&rkey) {
                out.push(entry.clone());
            } else {
                out.push((None, None));
            }
        }

        Ok(out)
    }

    /// Guarded Put: write the item maintaining OCC row version and ensuring no active 2PC lock.
    ///
    /// If `is_conditional` is true and `expected_version` is `Some(v)`, enforces that
    /// no active intent exists AND `m:v == v` using Bigtable's De Morgan form:
    /// `(active intent exists in m) OR (m:v != v)` with mutations placed in `false_mutations`.
    /// If `is_conditional` is true and `expected_version` is `None`, enforces that
    /// no active intent exists AND the row does not exist (`m:v` and family `d` do not exist).
    /// If `is_conditional` is false, only enforces that no active intent exists.
    /// In all cases, writes the item to family `d` and updates `m:v`.
    pub async fn put(
        &self,
        key_info: &TableKeyInfo,
        item: &Item,
        expected_version: Option<u64>,
        is_conditional: bool,
    ) -> Result<u64, StorageError> {
        let row_key = row_key::encode_key(item, &key_info.key_schema)?;
        let intent_timeout_micros = (self.intent_timeout_secs * 1_000_000) as i64;
        let now_micros = (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1000) as i64;
        let min_timestamp = now_micros - intent_timeout_micros;

        let next_version = expected_version.unwrap_or(0) + 1;
        let mut mutations = self.item_to_mutations(item, false)?;
        mutations.insert(0, Mutation {
            mutation: Some(mutation::Mutation::DeleteFromFamily(
                googleapis_tonic_google_bigtable_v2::google::bigtable::v2::mutation::DeleteFromFamily {
                    family_name: FAMILY_DATA.to_string(),
                }
            )),
        });
        mutations.push(Mutation {
            mutation: Some(mutation::Mutation::SetCell(SetCell {
                family_name: FAMILY_META.to_string(),
                column_qualifier: QUALIFIER_VERSION.to_vec(),
                timestamp_micros: -1,
                value: next_version.to_string().into_bytes(),
            })),
        });

        let mut data = self.client.data();

        if is_conditional {
            if let Some(v) = expected_version {
                // Updating existing row using De Morgan form:
                // Predicate matches if (active intent exists in m) OR (m:v != v).
                // Mutations are in false_mutations.
                let predicate = build_occ_conflict_filter(min_timestamp, v);

                let req = CheckAndMutateRowRequest {
                    table_name: self.full_table_name.clone(),
                    app_profile_id: self.client.app_profile_id.clone().unwrap_or_default(),
                    row_key,
                    predicate_filter: Some(predicate),
                    true_mutations: vec![],
                    false_mutations: mutations,
                    ..CheckAndMutateRowRequest::default()
                };

                let resp = data
                    .check_and_mutate_row(req)
                    .await
                    .map_err(|e| StorageError::Internal(format!("CheckAndMutateRow put: {e}")))?;

                if resp.predicate_matched {
                    return Err(StorageError::TransactionConflict(
                        "concurrent transaction holds an intent on this row or row version changed"
                            .to_string(),
                    ));
                }
            } else {
                // Inserting new row: fail if active intent exists OR row already exists.
                let predicate = build_row_presence_conflict_filter(min_timestamp);

                let req = CheckAndMutateRowRequest {
                    table_name: self.full_table_name.clone(),
                    app_profile_id: self.client.app_profile_id.clone().unwrap_or_default(),
                    row_key,
                    predicate_filter: Some(predicate),
                    true_mutations: vec![],
                    false_mutations: mutations,
                    ..CheckAndMutateRowRequest::default()
                };

                let resp = data
                    .check_and_mutate_row(req)
                    .await
                    .map_err(|e| StorageError::Internal(format!("CheckAndMutateRow put: {e}")))?;

                if resp.predicate_matched {
                    return Err(StorageError::TransactionConflict(
                        "concurrent transaction holds an intent on this row or row already exists"
                            .to_string(),
                    ));
                }
            }
        } else {
            // Unconditional write: only check for active 2PC intents.
            let predicate = build_active_intent_filter(min_timestamp);

            let req = CheckAndMutateRowRequest {
                table_name: self.full_table_name.clone(),
                app_profile_id: self.client.app_profile_id.clone().unwrap_or_default(),
                row_key,
                predicate_filter: Some(predicate),
                true_mutations: vec![],
                false_mutations: mutations,
                ..CheckAndMutateRowRequest::default()
            };

            let resp = data
                .check_and_mutate_row(req)
                .await
                .map_err(|e| StorageError::Internal(format!("CheckAndMutateRow put: {e}")))?;

            if resp.predicate_matched {
                return Err(StorageError::TransactionConflict(
                    "concurrent transaction holds an intent on this row".to_string(),
                ));
            }
        }

        Ok(next_version)
    }

    /// Guarded Delete: delete data only if condition passes and no active 2PC lock.
    /// Uses DeleteFromFamily(d) and DeleteFromColumn(m:v) to preserve intent cells.
    pub async fn delete(
        &self,
        key_info: &TableKeyInfo,
        key: &Item,
        expected_version: Option<u64>,
        is_conditional: bool,
    ) -> Result<(), StorageError> {
        let row_key = row_key::encode_key(key, &key_info.key_schema)?;
        let intent_timeout_micros = (self.intent_timeout_secs * 1_000_000) as i64;
        let now_micros = (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1000) as i64;
        let min_timestamp = now_micros - intent_timeout_micros;

        let mutations = vec![
            Mutation {
                mutation: Some(mutation::Mutation::DeleteFromFamily(
                    googleapis_tonic_google_bigtable_v2::google::bigtable::v2::mutation::DeleteFromFamily {
                        family_name: FAMILY_DATA.to_string(),
                    },
                )),
            },
            Mutation {
                mutation: Some(mutation::Mutation::DeleteFromColumn(
                    googleapis_tonic_google_bigtable_v2::google::bigtable::v2::mutation::DeleteFromColumn {
                        family_name: FAMILY_META.to_string(),
                        column_qualifier: QUALIFIER_VERSION.to_vec(),
                        time_range: None,
                    },
                )),
            },
        ];

        let mut data = self.client.data();

        if is_conditional {
            if let Some(v) = expected_version {
                // Deleting existing row using De Morgan form:
                // Predicate matches if (active intent exists in m) OR (m:v != v).
                // Mutations are in false_mutations.
                let predicate = build_occ_conflict_filter(min_timestamp, v);

                let req = CheckAndMutateRowRequest {
                    table_name: self.full_table_name.clone(),
                    app_profile_id: self.client.app_profile_id.clone().unwrap_or_default(),
                    row_key,
                    predicate_filter: Some(predicate),
                    true_mutations: vec![],
                    false_mutations: mutations,
                    ..CheckAndMutateRowRequest::default()
                };

                let resp = data.check_and_mutate_row(req).await.map_err(|e| {
                    StorageError::Internal(format!("CheckAndMutateRow delete: {e}"))
                })?;

                if resp.predicate_matched {
                    return Err(StorageError::TransactionConflict(
                        "concurrent transaction holds an intent on this row or row version changed"
                            .to_string(),
                    ));
                }
            } else {
                // Deleting non-existent row with condition: fail if active intent exists OR row exists.
                let predicate = build_row_presence_conflict_filter(min_timestamp);

                let req = CheckAndMutateRowRequest {
                    table_name: self.full_table_name.clone(),
                    app_profile_id: self.client.app_profile_id.clone().unwrap_or_default(),
                    row_key,
                    predicate_filter: Some(predicate),
                    true_mutations: vec![],
                    false_mutations: mutations,
                    ..CheckAndMutateRowRequest::default()
                };

                let resp = data.check_and_mutate_row(req).await.map_err(|e| {
                    StorageError::Internal(format!("CheckAndMutateRow delete: {e}"))
                })?;

                if resp.predicate_matched {
                    return Err(StorageError::TransactionConflict(
                        "concurrent transaction holds an intent on this row or row already exists"
                            .to_string(),
                    ));
                }
            }
        } else {
            // Unconditional delete: only check for active 2PC intents.
            let predicate = build_active_intent_filter(min_timestamp);

            let req = CheckAndMutateRowRequest {
                table_name: self.full_table_name.clone(),
                app_profile_id: self.client.app_profile_id.clone().unwrap_or_default(),
                row_key,
                predicate_filter: Some(predicate),
                true_mutations: vec![],
                false_mutations: mutations,
                ..CheckAndMutateRowRequest::default()
            };

            let resp = data
                .check_and_mutate_row(req)
                .await
                .map_err(|e| StorageError::Internal(format!("CheckAndMutateRow delete: {e}")))?;

            if resp.predicate_matched {
                return Err(StorageError::TransactionConflict(
                    "concurrent transaction holds an intent on this row".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Replace any existing row with the supplied item. Sets m:v. No condition check, no lock check.
    pub async fn put_unconditional(
        &self,
        key_info: &TableKeyInfo,
        item: &Item,
    ) -> Result<u64, StorageError> {
        let row_key = row_key::encode_key(item, &key_info.key_schema)?;
        let mut mutations = self.item_to_mutations(item, true)?;
        let now_micros = (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1000) as u64;
        mutations.push(Mutation {
            mutation: Some(mutation::Mutation::SetCell(SetCell {
                family_name: FAMILY_META.to_string(),
                column_qualifier: QUALIFIER_VERSION.to_vec(),
                timestamp_micros: -1,
                value: now_micros.to_string().into_bytes(),
            })),
        });
        let mut data = self.client.data();
        let req = MutateRowRequest {
            table_name: self.full_table_name.clone(),
            app_profile_id: self.client.app_profile_id.clone().unwrap_or_default(),
            row_key,
            mutations,
            ..MutateRowRequest::default()
        };
        data.mutate_row(req)
            .await
            .map_err(|e| StorageError::Internal(format!("MutateRow put: {e}")))?;
        Ok(now_micros)
    }

    /// Delete a row by key, removing data and m:v. No condition check, no lock check.
    pub async fn delete_unconditional(
        &self,
        key_info: &TableKeyInfo,
        key: &Item,
    ) -> Result<(), StorageError> {
        let row_key = row_key::encode_key(key, &key_info.key_schema)?;
        let mut data = self.client.data();
        let req = MutateRowRequest {
            table_name: self.full_table_name.clone(),
            app_profile_id: self.client.app_profile_id.clone().unwrap_or_default(),
            row_key,
            mutations: vec![
                Mutation {
                    mutation: Some(mutation::Mutation::DeleteFromFamily(
                        googleapis_tonic_google_bigtable_v2::google::bigtable::v2::mutation::DeleteFromFamily {
                            family_name: FAMILY_DATA.to_string(),
                        },
                    )),
                },
                Mutation {
                    mutation: Some(mutation::Mutation::DeleteFromColumn(
                        googleapis_tonic_google_bigtable_v2::google::bigtable::v2::mutation::DeleteFromColumn {
                            family_name: FAMILY_META.to_string(),
                            column_qualifier: QUALIFIER_VERSION.to_vec(),
                            time_range: None,
                        }
                    )),
                },
            ],
            ..MutateRowRequest::default()
        };
        data.mutate_row(req)
            .await
            .map_err(|e| StorageError::Internal(format!("MutateRow delete: {e}")))?;
        Ok(())
    }

    /// Apply a list of mutations (SetCell / DeleteFromColumn) to a row by key.
    pub async fn mutate_cells(
        &self,
        row_key_bytes: Vec<u8>,
        mutations: Vec<Mutation>,
    ) -> Result<(), StorageError> {
        let mut data = self.client.data();
        let req = MutateRowRequest {
            table_name: self.full_table_name.clone(),
            app_profile_id: self.client.app_profile_id.clone().unwrap_or_default(),
            row_key: row_key_bytes,
            mutations,
            ..MutateRowRequest::default()
        };
        data.mutate_row(req)
            .await
            .map_err(|e| StorageError::Internal(format!("MutateRow: {e}")))?;
        Ok(())
    }

    /// Turn an Item map into BigTable Mutations. When `delete_row_first` is
    /// true, prepend a DeleteFromFamily(d) so the new item fully replaces the old in data family.
    pub fn item_to_mutations(
        &self,
        item: &Item,
        delete_row_first: bool,
    ) -> Result<Vec<Mutation>, StorageError> {
        let mut mutations = Vec::with_capacity(item.len() + 1);
        if delete_row_first {
            mutations.push(Mutation {
                mutation: Some(mutation::Mutation::DeleteFromFamily(
                    googleapis_tonic_google_bigtable_v2::google::bigtable::v2::mutation::DeleteFromFamily {
                        family_name: FAMILY_DATA.to_string(),
                    },
                )),
            });
        }
        for (name, value) in item {
            mutations.push(Mutation {
                mutation: Some(mutation::Mutation::SetCell(SetCell {
                    family_name: FAMILY_DATA.to_string(),
                    column_qualifier: name.as_bytes().to_vec(),
                    timestamp_micros: -1,
                    value: cell::encode(value)?,
                })),
            });
        }
        Ok(mutations)
    }
}

/// Apply `UpdateAction`s to an existing item (or empty if absent). Returns the
/// new image. Used by UpdateItem.
pub fn apply_update_actions(
    existing: &Item,
    actions: &[UpdateAction],
    maps: &ExpressionMaps,
) -> Result<Item, StorageError> {
    let mut new_item = existing.clone();
    apply_update(actions, &mut new_item, maps)
        .map_err(|e| StorageError::Validation(format!("update expression evaluation: {e}")))?;
    Ok(new_item)
}

/// Evaluate a ConditionExpression against the existing item. Returns Ok(())
/// if the condition holds (or is None); ConditionFailed if it doesn't.
pub fn check_condition(
    existing: Option<&Item>,
    condition: Option<&Expr>,
    maps: &ExpressionMaps,
) -> Result<(), StorageError> {
    let Some(expr) = condition else {
        return Ok(());
    };
    let empty: Item = BTreeMap::new();
    let item = existing.unwrap_or(&empty);
    match evaluate_condition(expr, item, maps) {
        Ok(true) => Ok(()),
        Ok(false) => Err(StorageError::ConditionFailed(existing.cloned())),
        Err(e) => Err(StorageError::Validation(format!(
            "condition evaluation: {e}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_version_string_and_bytes() {
        assert_eq!(decode_version(b"1"), Some(1));
        assert_eq!(decode_version(b"42"), Some(42));
        assert_eq!(decode_version(b"18446744073709551615"), Some(u64::MAX));

        let be_bytes = 123456789012345u64.to_be_bytes();
        assert_eq!(decode_version(&be_bytes), Some(123456789012345));

        assert_eq!(decode_version(b"not_a_number"), None);
        assert_eq!(decode_version(b""), None);
    }

    #[test]
    fn test_filter_builders() {
        let intent_filter = build_active_intent_filter(1000);
        assert!(intent_filter.filter.is_some());

        let version_filter = build_version_match_filter(42);
        assert!(version_filter.filter.is_some());

        let mismatch_filter = build_version_mismatch_filter(42);
        assert!(mismatch_filter.filter.is_some());

        let occ_conflict_filter = build_occ_conflict_filter(1000, 42);
        assert!(occ_conflict_filter.filter.is_some());

        let conflict_filter = build_row_presence_conflict_filter(1000);
        assert!(conflict_filter.filter.is_some());
    }

    #[test]
    fn test_version_match_filter_order() {
        let filter = build_version_match_filter(42);
        if let Some(Filter::Chain(chain)) = filter.filter {
            assert_eq!(chain.filters.len(), 4);
            // 0: FamilyNameRegexFilter
            assert!(matches!(
                chain.filters[0].filter,
                Some(Filter::FamilyNameRegexFilter(_))
            ));
            // 1: ColumnQualifierRegexFilter
            assert!(matches!(
                chain.filters[1].filter,
                Some(Filter::ColumnQualifierRegexFilter(_))
            ));
            // 2: MUST be CellsPerColumnLimitFilter(1) before ValueRegexFilter to prevent matching stale history!
            assert!(matches!(
                chain.filters[2].filter,
                Some(Filter::CellsPerColumnLimitFilter(1))
            ));
            // 3: ValueRegexFilter
            assert!(matches!(
                chain.filters[3].filter,
                Some(Filter::ValueRegexFilter(_))
            ));
        } else {
            panic!("expected Filter::Chain");
        }
    }
}
