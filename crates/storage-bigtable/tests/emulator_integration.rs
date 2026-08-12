// Copyright 2026 ExtendDB contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration test suite for Google Cloud Bigtable storage backend.
//!
//! Validates:
//! - Catalog bootstrapping and table lifecycle.
//! - Guarded single-row CRUD operations (PutItem, GetItem, UpdateItem, DeleteItem) with expression parsing and arithmetic.
//! - 2PC multi-row transactions (TransactWriteItems), atomic rollback, idempotency replay/mismatch, and concurrent TransactGetItems.
//! - Decimal number key encoding, exact numerical sorting, and forward/reverse range scans.
//! - GSI shadow table operations, range queries, update propagation, and deletion cleanup.
//! - TTL index maintenance and background sweep.
//!
//! These tests run when `BIGTABLE_EMULATOR_HOST` is set, and skip gracefully otherwise.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use extenddb_core::expression::{
    Expr, ExpressionMaps, KeyCondition, UpdateAction,
    parse_condition, parse_key_condition, parse_update, tokenize,
};
use extenddb_core::types::{
    AttributeDefinition, AttributeValue, BillingMode, CreateTableInput,
    DescribeTableInput, GsiInput, Item, KeySchemaElement, KeyType,
    ListTablesInput, Projection, ProjectionType,
    ReturnValuesOnConditionCheckFailure, ScalarAttributeType,
    TableStatus, TimeToLiveStatus,
};
use extenddb_storage::bootstrapper::Bootstrapper;
use extenddb_storage::error::StorageError;
use extenddb_storage::{
    DataEngine, IdempotencyKey, MetadataEngine, TableEngine, TransactGetOp, TransactWriteOp,
};
use extenddb_storage_bigtable::data::admin::AdminClient;
use extenddb_storage_bigtable::transact::ensure_txn_log_table;
use extenddb_storage_bigtable::ttl_worker::{ensure_ttl_index_table, sweep_once};
use extenddb_storage_bigtable::{
    BigtableBootstrapper, BigtableClient, BigtableEngine, BigtableStorageConfig,
};
use uuid::Uuid;

struct TestContext {
    config: BigtableStorageConfig,
    client: Arc<BigtableClient>,
    engine: Arc<BigtableEngine>,
    account_id: String,
}

fn unique_name(prefix: &str) -> String {
    format!("{}_{}", prefix, &Uuid::new_v4().to_string().replace('-', "")[..12])
}

fn parse_cond(expr_str: &str) -> Expr {
    let tokens = tokenize(expr_str).expect("tokenize condition");
    parse_condition(&tokens).expect("parse condition")
}

fn parse_upd(expr_str: &str) -> Vec<UpdateAction> {
    let tokens = tokenize(expr_str).expect("tokenize update");
    parse_update(&tokens).expect("parse update")
}

fn parse_kc(expr_str: &str) -> KeyCondition {
    let tokens = tokenize(expr_str).expect("tokenize key condition");
    parse_key_condition(&tokens).expect("parse key condition")
}

async fn setup_emulator_context() -> Option<TestContext> {
    let host = match std::env::var("BIGTABLE_EMULATOR_HOST") {
        Ok(h) if !h.trim().is_empty() => h,
        _ => return None,
    };

    let suffix = &Uuid::new_v4().to_string().replace('-', "")[..8];
    let project_id = std::env::var("BIGTABLE_PROJECT_ID")
        .unwrap_or_else(|_| format!("test-proj-{suffix}"));
    let instance_id = std::env::var("BIGTABLE_INSTANCE_ID")
        .unwrap_or_else(|_| format!("test-inst-{suffix}"));

    let config = BigtableStorageConfig {
        project_id,
        instance_id,
        emulator_host: Some(host),
        dev_mode: true,
        ..BigtableStorageConfig::default()
    };

    let client = match BigtableClient::connect(&config).await {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Skipping Bigtable emulator integration test (connect failed): {e}");
            return None;
        }
    };

    // Active probe to verify the emulator endpoint is reachable
    let mut admin = match AdminClient::connect(&client).await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Skipping Bigtable emulator integration test (admin connect failed): {e}");
            return None;
        }
    };
    if let Err(e) = admin.list_tables().await {
        eprintln!("Skipping Bigtable emulator integration test (list_tables probe failed): {e}");
        return None;
    }

    // Bootstrap catalog and data infrastructure
    let bootstrapper = BigtableBootstrapper::new(config.clone());
    let _ = bootstrapper.create_catalog_db().await;
    let _ = bootstrapper.create_data_db().await;
    let _ = bootstrapper.run_catalog_migrations().await;
    let _ = bootstrapper.bootstrap_encryption_key().await;
    let _ = bootstrapper.bootstrap_default_account().await;
    let _ = bootstrapper.bootstrap_admin_user(None, None).await;

    let _ = ensure_txn_log_table(&client).await;
    let _ = ensure_ttl_index_table(&client).await;

    let engine = Arc::new(BigtableEngine::new(
        client.clone(),
        client.clone(),
        config.intent_timeout_secs,
    ));
    let account_id = "123456789012".to_string();

    Some(TestContext {
        config,
        client,
        engine,
        account_id,
    })
}

// -----------------------------------------------------------------------------
// Test 1: Bootstrapping & Table Lifecycle
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_bootstrapping_and_table_lifecycle() {
    let Some(ctx) = setup_emulator_context().await else {
        eprintln!("Skipping Bigtable test: emulator unavailable");
        return;
    };

    let bootstrapper = BigtableBootstrapper::new(ctx.config.clone());
    let init_res = bootstrapper.is_catalog_initialized().await;
    assert!(init_res.is_ok(), "is_catalog_initialized failed: {:?}", init_res.err());
    assert!(init_res.unwrap(), "catalog should be initialized");

    let version = bootstrapper.read_catalog_version().await;
    assert!(version.is_ok(), "read_catalog_version failed: {:?}", version.err());
    assert_eq!(version.unwrap(), Some("0.1.0".to_string()));

    let table_name = unique_name("tbl_lifecycle");

    let input = CreateTableInput {
        table_name: table_name.clone(),
        key_schema: vec![
            KeySchemaElement {
                attribute_name: "pk".to_string(),
                key_type: KeyType::Hash,
            },
            KeySchemaElement {
                attribute_name: "sk".to_string(),
                key_type: KeyType::Range,
            },
        ],
        attribute_definitions: vec![
            AttributeDefinition {
                attribute_name: "pk".to_string(),
                attribute_type: ScalarAttributeType::S,
            },
            AttributeDefinition {
                attribute_name: "sk".to_string(),
                attribute_type: ScalarAttributeType::S,
            },
        ],
        billing_mode: Some(BillingMode::PayPerRequest),
        provisioned_throughput: None,
        global_secondary_indexes: None,
        local_secondary_indexes: None,
        stream_specification: None,
        sse_specification: None,
        tags: None,
        deletion_protection_enabled: Some(false),
        table_class: None,
        on_demand_throughput: None,
    };

    let desc = ctx
        .engine
        .create_table(&ctx.account_id, input)
        .await
        .expect("create_table failed");
    assert_eq!(desc.table_name, table_name);
    assert_eq!(desc.table_status, TableStatus::Active);

    // Describe table
    let described = ctx
        .engine
        .describe_table(&ctx.account_id, DescribeTableInput { table_name: table_name.clone() })
        .await
        .expect("describe_table failed");
    assert_eq!(described.table_name, table_name);
    assert_eq!(described.key_schema.len(), 2);

    // List tables
    let list = ctx
        .engine
        .list_tables(&ctx.account_id, ListTablesInput { limit: None, exclusive_start_table_name: None })
        .await
        .expect("list_tables failed");
    assert!(
        list.table_names.contains(&table_name),
        "created table should appear in list_tables"
    );

    // TableKeyInfo
    let key_info = ctx
        .engine
        .table_key_info(&ctx.account_id, &table_name)
        .await
        .expect("table_key_info failed");
    assert_eq!(key_info.table_name, table_name);
    assert_eq!(key_info.key_schema.len(), 2);
}

// -----------------------------------------------------------------------------
// Test 2: Guarded Single-Row CRUD Operations
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_guarded_single_row_crud() {
    let Some(ctx) = setup_emulator_context().await else {
        eprintln!("Skipping Bigtable test: emulator unavailable");
        return;
    };

    let table_name = unique_name("tbl_crud");

    let input = CreateTableInput {
        table_name: table_name.clone(),
        key_schema: vec![
            KeySchemaElement {
                attribute_name: "pk".to_string(),
                key_type: KeyType::Hash,
            },
            KeySchemaElement {
                attribute_name: "sk".to_string(),
                key_type: KeyType::Range,
            },
        ],
        attribute_definitions: vec![
            AttributeDefinition {
                attribute_name: "pk".to_string(),
                attribute_type: ScalarAttributeType::S,
            },
            AttributeDefinition {
                attribute_name: "sk".to_string(),
                attribute_type: ScalarAttributeType::S,
            },
        ],
        billing_mode: Some(BillingMode::PayPerRequest),
        provisioned_throughput: None,
        global_secondary_indexes: None,
        local_secondary_indexes: None,
        stream_specification: None,
        sse_specification: None,
        tags: None,
        deletion_protection_enabled: Some(false),
        table_class: None,
        on_demand_throughput: None,
    };

    let _ = ctx
        .engine
        .create_table(&ctx.account_id, input)
        .await
        .expect("create_table failed");
    let key_info = ctx
        .engine
        .table_key_info(&ctx.account_id, &table_name)
        .await
        .expect("table_key_info failed");

    // 1. PutItem (unconditional)
    let mut item: Item = BTreeMap::new();
    item.insert("pk".to_string(), AttributeValue::S("user#1".to_string()));
    item.insert("sk".to_string(), AttributeValue::S("profile".to_string()));
    item.insert("name".to_string(), AttributeValue::S("Alice".to_string()));
    item.insert("age".to_string(), AttributeValue::N("30".to_string()));
    item.insert("counter".to_string(), AttributeValue::N("100".to_string()));
    item.insert("active".to_string(), AttributeValue::Bool(true));
    item.insert(
        "tags".to_string(),
        AttributeValue::SS(BTreeSet::from(["admin".to_string(), "eng".to_string()])),
    );

    let empty_maps = ExpressionMaps::default();
    let old = ctx
        .engine
        .put_item(&key_info, item.clone(), false, None, &empty_maps, None)
        .await
        .expect("put_item failed");
    assert!(old.is_none());

    // 2. GetItem
    let mut key: Item = BTreeMap::new();
    key.insert("pk".to_string(), AttributeValue::S("user#1".to_string()));
    key.insert("sk".to_string(), AttributeValue::S("profile".to_string()));

    let fetched = ctx
        .engine
        .get_item(&key_info, &key)
        .await
        .expect("get_item failed")
        .expect("item not found");

    assert_eq!(fetched.get("name"), Some(&AttributeValue::S("Alice".to_string())));
    assert_eq!(fetched.get("age"), Some(&AttributeValue::N("30".to_string())));
    assert_eq!(fetched.get("counter"), Some(&AttributeValue::N("100".to_string())));
    assert_eq!(fetched.get("active"), Some(&AttributeValue::Bool(true)));

    // 3. PutItem (guarded with failing condition)
    let cond = parse_cond("attribute_not_exists(pk)");
    let ccf_result = ctx
        .engine
        .put_item(&key_info, item.clone(), false, Some(&cond), &empty_maps, None)
        .await;
    assert!(
        matches!(ccf_result, Err(StorageError::ConditionFailed(_))),
        "expected ConditionFailed, got {:?}",
        ccf_result
    );

    // 4. UpdateItem with arithmetic action and condition check
    let mut update_maps = ExpressionMaps::default();
    update_maps
        .values
        .insert("new_age".to_string(), AttributeValue::N("31".to_string()));
    update_maps
        .values
        .insert("email".to_string(), AttributeValue::S("alice@example.com".to_string()));
    update_maps
        .values
        .insert("inc".to_string(), AttributeValue::N("25".to_string()));
    update_maps
        .values
        .insert("expected_age".to_string(), AttributeValue::N("30".to_string()));

    let update_actions = parse_upd("SET age = :new_age, email = :email, counter = counter + :inc REMOVE active");
    let update_cond = parse_cond("age = :expected_age");

    let (old_upd, new_upd) = ctx
        .engine
        .update_item(
            &key_info,
            &key,
            &update_actions,
            true,
            true,
            Some(&update_cond),
            &update_maps,
            None,
        )
        .await
        .expect("guarded update_item failed");

    let old_upd = old_upd.expect("expected old item");
    assert_eq!(old_upd.get("age"), Some(&AttributeValue::N("30".to_string())));

    let new_upd = new_upd.expect("expected new item");
    assert_eq!(new_upd.get("age"), Some(&AttributeValue::N("31".to_string())));
    assert_eq!(new_upd.get("counter"), Some(&AttributeValue::N("125".to_string())));
    assert_eq!(
        new_upd.get("email"),
        Some(&AttributeValue::S("alice@example.com".to_string()))
    );
    assert!(!new_upd.contains_key("active"));

    // 5. UpdateItem with failing condition (now age is 31, :expected_age is 30)
    let failing_cond = parse_cond("age = :expected_age");
    let upd_fail = ctx
        .engine
        .update_item(
            &key_info,
            &key,
            &update_actions,
            false,
            false,
            Some(&failing_cond),
            &update_maps,
            None,
        )
        .await;
    assert!(
        matches!(upd_fail, Err(StorageError::ConditionFailed(_))),
        "expected ConditionFailed on update"
    );

    // 6. DeleteItem with failing condition
    let del_fail = ctx
        .engine
        .delete_item(
            &key_info,
            &key,
            false,
            Some(&failing_cond),
            &update_maps,
            None,
        )
        .await;
    assert!(
        matches!(del_fail, Err(StorageError::ConditionFailed(_))),
        "expected ConditionFailed on delete"
    );

    // 7. DeleteItem (unconditional)
    let del_ok = ctx
        .engine
        .delete_item(&key_info, &key, true, None, &empty_maps, None)
        .await
        .expect("delete_item failed");
    assert!(del_ok.is_some());

    // Verify deleted
    let post_delete = ctx
        .engine
        .get_item(&key_info, &key)
        .await
        .expect("get_item failed");
    assert!(post_delete.is_none());
}

// -----------------------------------------------------------------------------
// Test 3: 2PC Multi-Row Transactions, Rollback & Idempotency
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_transact_write_items_and_rollback() {
    let Some(ctx) = setup_emulator_context().await else {
        eprintln!("Skipping Bigtable test: emulator unavailable");
        return;
    };

    let tbl1_name = unique_name("tbl_tx_orders");
    let tbl2_name = unique_name("tbl_tx_inventory");

    // Table 1 (Orders)
    let input1 = CreateTableInput {
        table_name: tbl1_name.clone(),
        key_schema: vec![KeySchemaElement {
            attribute_name: "order_id".to_string(),
            key_type: KeyType::Hash,
        }],
        attribute_definitions: vec![AttributeDefinition {
            attribute_name: "order_id".to_string(),
            attribute_type: ScalarAttributeType::S,
        }],
        billing_mode: Some(BillingMode::PayPerRequest),
        provisioned_throughput: None,
        global_secondary_indexes: None,
        local_secondary_indexes: None,
        stream_specification: None,
        sse_specification: None,
        tags: None,
        deletion_protection_enabled: Some(false),
        table_class: None,
        on_demand_throughput: None,
    };
    let _ = ctx.engine.create_table(&ctx.account_id, input1).await.unwrap();
    let key_info1 = ctx.engine.table_key_info(&ctx.account_id, &tbl1_name).await.unwrap();

    // Table 2 (Inventory)
    let input2 = CreateTableInput {
        table_name: tbl2_name.clone(),
        key_schema: vec![KeySchemaElement {
            attribute_name: "item_id".to_string(),
            key_type: KeyType::Hash,
        }],
        attribute_definitions: vec![AttributeDefinition {
            attribute_name: "item_id".to_string(),
            attribute_type: ScalarAttributeType::S,
        }],
        billing_mode: Some(BillingMode::PayPerRequest),
        provisioned_throughput: None,
        global_secondary_indexes: None,
        local_secondary_indexes: None,
        stream_specification: None,
        sse_specification: None,
        tags: None,
        deletion_protection_enabled: Some(false),
        table_class: None,
        on_demand_throughput: None,
    };
    let _ = ctx.engine.create_table(&ctx.account_id, input2).await.unwrap();
    let key_info2 = ctx.engine.table_key_info(&ctx.account_id, &tbl2_name).await.unwrap();

    let empty_maps = ExpressionMaps::default();

    // 1. Successful 2PC transaction across distinct tables with idempotency token
    let mut order_item: Item = BTreeMap::new();
    order_item.insert("order_id".to_string(), AttributeValue::S("ord#1001".to_string()));
    order_item.insert("total".to_string(), AttributeValue::N("250".to_string()));

    let mut inv_item: Item = BTreeMap::new();
    inv_item.insert("item_id".to_string(), AttributeValue::S("prod#50".to_string()));
    inv_item.insert("stock".to_string(), AttributeValue::N("99".to_string()));

    let ops = vec![
        TransactWriteOp::Put {
            key_info: &key_info1,
            item: &order_item,
            condition: None,
            maps: &empty_maps,
            return_values_on_ccf: ReturnValuesOnConditionCheckFailure::None,
            stream: None,
        },
        TransactWriteOp::Put {
            key_info: &key_info2,
            item: &inv_item,
            condition: None,
            maps: &empty_maps,
            return_values_on_ccf: ReturnValuesOnConditionCheckFailure::None,
            stream: None,
        },
    ];

    let token_str = format!("tok-{}", Uuid::new_v4());
    let token = IdempotencyKey {
        account_id: &ctx.account_id,
        token: &token_str,
        fingerprint: "fp-ord1001",
    };

    ctx.engine
        .transact_write_items(&ops, Some(token))
        .await
        .expect("transact_write_items across distinct tables failed");

    // Verify both writes landed
    let mut ord_key: Item = BTreeMap::new();
    ord_key.insert("order_id".to_string(), AttributeValue::S("ord#1001".to_string()));
    let ord_get = ctx.engine.get_item(&key_info1, &ord_key).await.unwrap();
    assert!(ord_get.is_some());

    let mut inv_key: Item = BTreeMap::new();
    inv_key.insert("item_id".to_string(), AttributeValue::S("prod#50".to_string()));
    let inv_get = ctx.engine.get_item(&key_info2, &inv_key).await.unwrap();
    assert!(inv_get.is_some());

    // 2. Test Idempotent Replay and Mismatch
    let replay_token = IdempotencyKey {
        account_id: &ctx.account_id,
        token: &token_str,
        fingerprint: "fp-ord1001",
    };
    let replay_res = ctx.engine.transact_write_items(&[], Some(replay_token)).await;
    assert!(matches!(replay_res, Err(StorageError::IdempotentReplay)));

    let mismatch_token = IdempotencyKey {
        account_id: &ctx.account_id,
        token: &token_str,
        fingerprint: "fp-mismatch",
    };
    let mismatch_res = ctx.engine.transact_write_items(&[], Some(mismatch_token)).await;
    assert!(matches!(mismatch_res, Err(StorageError::IdempotentMismatch)));

    // 3. Concurrent TransactGetItems across both tables
    let get_op1 = TransactGetOp {
        key_info: &key_info1,
        key: &ord_key,
    };
    let get_op2 = TransactGetOp {
        key_info: &key_info2,
        key: &inv_key,
    };
    let t_get_res = ctx.engine.transact_get_items(&[get_op1, get_op2]).await;
    assert!(t_get_res.is_ok());
    let fetched_items = t_get_res.unwrap();
    assert_eq!(fetched_items.len(), 2);
    assert!(fetched_items[0].is_some());
    assert!(fetched_items[1].is_some());

    // 4. Transaction Rollback on Condition Check Failure
    let mut order_item2: Item = BTreeMap::new();
    order_item2.insert("order_id".to_string(), AttributeValue::S("ord#1002".to_string()));
    order_item2.insert("total".to_string(), AttributeValue::N("500".to_string()));

    let mut missing_inv_key: Item = BTreeMap::new();
    missing_inv_key.insert("item_id".to_string(), AttributeValue::S("prod#9999".to_string()));

    let fail_cond = parse_cond("attribute_exists(item_id)");

    let rollback_ops = vec![
        TransactWriteOp::Put {
            key_info: &key_info1,
            item: &order_item2,
            condition: None,
            maps: &empty_maps,
            return_values_on_ccf: ReturnValuesOnConditionCheckFailure::None,
            stream: None,
        },
        TransactWriteOp::ConditionCheck {
            key_info: &key_info2,
            key: &missing_inv_key,
            condition: &fail_cond,
            maps: &empty_maps,
            return_values_on_ccf: ReturnValuesOnConditionCheckFailure::None,
        },
    ];

    let tx_fail_res = ctx.engine.transact_write_items(&rollback_ops, None).await;
    assert!(
        tx_fail_res.is_err(),
        "transaction should fail due to condition check failure"
    );

    // Verify order_item2 was NOT written (atomicity rollback verified!)
    let mut ord2_key: Item = BTreeMap::new();
    ord2_key.insert("order_id".to_string(), AttributeValue::S("ord#1002".to_string()));
    let ord2_get = ctx.engine.get_item(&key_info1, &ord2_key).await.unwrap();
    assert!(
        ord2_get.is_none(),
        "order_item2 must have been rolled back and not exist in table"
    );
}

// -----------------------------------------------------------------------------
// Test 4: Decimal Number Key Encoding & Range Scans
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_decimal_number_keys_and_range_scan() {
    let Some(ctx) = setup_emulator_context().await else {
        eprintln!("Skipping Bigtable test: emulator unavailable");
        return;
    };

    let table_name = unique_name("tbl_decimal_keys");

    let input = CreateTableInput {
        table_name: table_name.clone(),
        key_schema: vec![
            KeySchemaElement {
                attribute_name: "sensor_id".to_string(),
                key_type: KeyType::Hash,
            },
            KeySchemaElement {
                attribute_name: "reading".to_string(),
                key_type: KeyType::Range,
            },
        ],
        attribute_definitions: vec![
            AttributeDefinition {
                attribute_name: "sensor_id".to_string(),
                attribute_type: ScalarAttributeType::S,
            },
            AttributeDefinition {
                attribute_name: "reading".to_string(),
                attribute_type: ScalarAttributeType::N,
            },
        ],
        billing_mode: Some(BillingMode::PayPerRequest),
        provisioned_throughput: None,
        global_secondary_indexes: None,
        local_secondary_indexes: None,
        stream_specification: None,
        sse_specification: None,
        tags: None,
        deletion_protection_enabled: Some(false),
        table_class: None,
        on_demand_throughput: None,
    };

    let _ = ctx.engine.create_table(&ctx.account_id, input).await.unwrap();
    let key_info = ctx.engine.table_key_info(&ctx.account_id, &table_name).await.unwrap();

    let readings = vec![
        "-1000.5", "-100.5", "-50.25", "-1.0", "0", "0.005", "0.5", "1.0", "10.5", "100.0", "999.99", "1000.5",
    ];

    let empty_maps = ExpressionMaps::default();
    for r in &readings {
        let mut item: Item = BTreeMap::new();
        item.insert("sensor_id".to_string(), AttributeValue::S("sensor#1".to_string()));
        item.insert("reading".to_string(), AttributeValue::N((*r).to_string()));
        item.insert("status".to_string(), AttributeValue::S("OK".to_string()));
        ctx.engine
            .put_item(&key_info, item, false, None, &empty_maps, None)
            .await
            .unwrap();
    }

    // Range Query: reading BETWEEN -50.25 AND 10.5
    let mut query_maps = ExpressionMaps::default();
    query_maps
        .values
        .insert("sid".to_string(), AttributeValue::S("sensor#1".to_string()));
    query_maps
        .values
        .insert("lo".to_string(), AttributeValue::N("-50.25".to_string()));
    query_maps
        .values
        .insert("hi".to_string(), AttributeValue::N("10.5".to_string()));

    let kc = parse_kc("sensor_id = :sid AND reading BETWEEN :lo AND :hi");

    let (items, _) = ctx
        .engine
        .query(
            &key_info,
            &kc,
            &query_maps,
            /* forward */ true,
            /* limit */ None,
            /* exclusive_start_key */ None,
            /* index_name */ None,
        )
        .await
        .expect("query failed");

    let expected = vec!["-50.25", "-1.0", "0", "0.005", "0.5", "1.0", "10.5"];
    let actual: Vec<String> = items
        .iter()
        .map(|item| match item.get("reading").unwrap() {
            AttributeValue::N(s) => s.clone(),
            _ => panic!("expected N attribute"),
        })
        .collect();

    assert_eq!(actual, expected, "readings must match numerical sort order");

    // Reverse scan
    let (rev_items, _) = ctx
        .engine
        .query(
            &key_info,
            &kc,
            &query_maps,
            /* forward */ false,
            /* limit */ None,
            /* exclusive_start_key */ None,
            /* index_name */ None,
        )
        .await
        .expect("reverse query failed");

    let mut rev_expected = expected.clone();
    rev_expected.reverse();
    let rev_actual: Vec<String> = rev_items
        .iter()
        .map(|item| match item.get("reading").unwrap() {
            AttributeValue::N(s) => s.clone(),
            _ => panic!("expected N attribute"),
        })
        .collect();

    assert_eq!(rev_actual, rev_expected, "reverse scan must match reversed numerical order");

    // Scan table with limit pagination
    let (scanned_first, last_key) = ctx
        .engine
        .scan(&key_info, Some(6), None, None, None, None)
        .await
        .expect("scan first page");
    assert_eq!(scanned_first.len(), 6);
    assert!(last_key.is_some());

    let (scanned_second, _) = ctx
        .engine
        .scan(&key_info, Some(10), last_key.as_ref(), None, None, None)
        .await
        .expect("scan second page");
    assert_eq!(scanned_second.len(), 6);
}

// -----------------------------------------------------------------------------
// Test 5: GSI Shadow Table Operations & Queries
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_gsi_shadow_table_operations_and_query() {
    let Some(ctx) = setup_emulator_context().await else {
        eprintln!("Skipping Bigtable test: emulator unavailable");
        return;
    };

    let table_name = unique_name("tbl_gsi_test");
    let index_name = "DepartmentSalaryIndex";

    let gsi = GsiInput {
        index_name: index_name.to_string(),
        key_schema: vec![
            KeySchemaElement {
                attribute_name: "department".to_string(),
                key_type: KeyType::Hash,
            },
            KeySchemaElement {
                attribute_name: "salary".to_string(),
                key_type: KeyType::Range,
            },
        ],
        projection: Projection {
            projection_type: ProjectionType::All,
            non_key_attributes: None,
        },
        provisioned_throughput: None,
    };

    let input = CreateTableInput {
        table_name: table_name.clone(),
        key_schema: vec![KeySchemaElement {
            attribute_name: "emp_id".to_string(),
            key_type: KeyType::Hash,
        }],
        attribute_definitions: vec![
            AttributeDefinition {
                attribute_name: "emp_id".to_string(),
                attribute_type: ScalarAttributeType::S,
            },
            AttributeDefinition {
                attribute_name: "department".to_string(),
                attribute_type: ScalarAttributeType::S,
            },
            AttributeDefinition {
                attribute_name: "salary".to_string(),
                attribute_type: ScalarAttributeType::N,
            },
        ],
        billing_mode: Some(BillingMode::PayPerRequest),
        provisioned_throughput: None,
        global_secondary_indexes: Some(vec![gsi]),
        local_secondary_indexes: None,
        stream_specification: None,
        sse_specification: None,
        tags: None,
        deletion_protection_enabled: Some(false),
        table_class: None,
        on_demand_throughput: None,
    };

    let _ = ctx.engine.create_table(&ctx.account_id, input).await.unwrap();
    let key_info = ctx.engine.table_key_info(&ctx.account_id, &table_name).await.unwrap();

    let empty_maps = ExpressionMaps::default();

    // Put employees
    let mut emp1: Item = BTreeMap::new();
    emp1.insert("emp_id".to_string(), AttributeValue::S("emp#1".to_string()));
    emp1.insert("department".to_string(), AttributeValue::S("Engineering".to_string()));
    emp1.insert("salary".to_string(), AttributeValue::N("120000".to_string()));
    emp1.insert("name".to_string(), AttributeValue::S("Alice".to_string()));
    ctx.engine.put_item(&key_info, emp1, false, None, &empty_maps, None).await.unwrap();

    let mut emp2: Item = BTreeMap::new();
    emp2.insert("emp_id".to_string(), AttributeValue::S("emp#2".to_string()));
    emp2.insert("department".to_string(), AttributeValue::S("Engineering".to_string()));
    emp2.insert("salary".to_string(), AttributeValue::N("150000".to_string()));
    emp2.insert("name".to_string(), AttributeValue::S("Bob".to_string()));
    ctx.engine.put_item(&key_info, emp2, false, None, &empty_maps, None).await.unwrap();

    let mut emp3: Item = BTreeMap::new();
    emp3.insert("emp_id".to_string(), AttributeValue::S("emp#3".to_string()));
    emp3.insert("department".to_string(), AttributeValue::S("Sales".to_string()));
    emp3.insert("salary".to_string(), AttributeValue::N("90000".to_string()));
    emp3.insert("name".to_string(), AttributeValue::S("Carol".to_string()));
    ctx.engine.put_item(&key_info, emp3, false, None, &empty_maps, None).await.unwrap();

    // Query GSI on department = "Engineering"
    let mut gsi_maps = ExpressionMaps::default();
    gsi_maps
        .values
        .insert("dept".to_string(), AttributeValue::S("Engineering".to_string()));

    let gsi_kc = parse_kc("department = :dept");

    let (gsi_items, _) = ctx
        .engine
        .query(
            &key_info,
            &gsi_kc,
            &gsi_maps,
            /* forward */ true,
            /* limit */ None,
            /* exclusive_start_key */ None,
            /* index_name */ Some(index_name),
        )
        .await
        .expect("GSI query failed");

    assert_eq!(gsi_items.len(), 2);
    assert_eq!(
        gsi_items[0].get("emp_id"),
        Some(&AttributeValue::S("emp#1".to_string()))
    );
    assert_eq!(
        gsi_items[1].get("emp_id"),
        Some(&AttributeValue::S("emp#2".to_string()))
    );

    // Update emp1 to department = "Management" and salary = 175000
    let mut upd_key: Item = BTreeMap::new();
    upd_key.insert("emp_id".to_string(), AttributeValue::S("emp#1".to_string()));

    let mut upd_maps = ExpressionMaps::default();
    upd_maps.values.insert(
        "new_dept".to_string(),
        AttributeValue::S("Management".to_string()),
    );
    upd_maps.values.insert(
        "new_salary".to_string(),
        AttributeValue::N("175000".to_string()),
    );
    let upd_actions = parse_upd("SET department = :new_dept, salary = :new_salary");

    ctx.engine
        .update_item(
            &key_info,
            &upd_key,
            &upd_actions,
            false,
            false,
            None,
            &upd_maps,
            None,
        )
        .await
        .expect("update emp1 failed");

    // Re-query GSI for department = "Engineering" -> now only emp2
    let (gsi_items_after, _) = ctx
        .engine
        .query(
            &key_info,
            &gsi_kc,
            &gsi_maps,
            true,
            None,
            None,
            Some(index_name),
        )
        .await
        .unwrap();
    assert_eq!(gsi_items_after.len(), 1);
    assert_eq!(
        gsi_items_after[0].get("emp_id"),
        Some(&AttributeValue::S("emp#2".to_string()))
    );

    // Query GSI for department = "Management" -> emp1
    let mut mgmt_maps = ExpressionMaps::default();
    mgmt_maps
        .values
        .insert("dept".to_string(), AttributeValue::S("Management".to_string()));
    let (mgmt_items, _) = ctx
        .engine
        .query(
            &key_info,
            &gsi_kc,
            &mgmt_maps,
            true,
            None,
            None,
            Some(index_name),
        )
        .await
        .unwrap();
    assert_eq!(mgmt_items.len(), 1);
    assert_eq!(
        mgmt_items[0].get("emp_id"),
        Some(&AttributeValue::S("emp#1".to_string()))
    );

    // Delete emp2 -> GSI shadow entry should also be cleaned up
    let mut del_key: Item = BTreeMap::new();
    del_key.insert("emp_id".to_string(), AttributeValue::S("emp#2".to_string()));

    ctx.engine
        .delete_item(&key_info, &del_key, false, None, &empty_maps, None)
        .await
        .expect("delete emp2 failed");

    let (gsi_items_after_del, _) = ctx
        .engine
        .query(
            &key_info,
            &gsi_kc,
            &gsi_maps,
            true,
            None,
            None,
            Some(index_name),
        )
        .await
        .unwrap();
    assert_eq!(gsi_items_after_del.len(), 0);
}

// -----------------------------------------------------------------------------
// Test 6: TTL Index Maintenance & Sweep
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_ttl_index_operations() {
    let Some(ctx) = setup_emulator_context().await else {
        eprintln!("Skipping Bigtable test: emulator unavailable");
        return;
    };

    let table_name = unique_name("tbl_ttl");

    let input = CreateTableInput {
        table_name: table_name.clone(),
        key_schema: vec![KeySchemaElement {
            attribute_name: "session_id".to_string(),
            key_type: KeyType::Hash,
        }],
        attribute_definitions: vec![AttributeDefinition {
            attribute_name: "session_id".to_string(),
            attribute_type: ScalarAttributeType::S,
        }],
        billing_mode: Some(BillingMode::PayPerRequest),
        provisioned_throughput: None,
        global_secondary_indexes: None,
        local_secondary_indexes: None,
        stream_specification: None,
        sse_specification: None,
        tags: None,
        deletion_protection_enabled: Some(false),
        table_class: None,
        on_demand_throughput: None,
    };

    let _ = ctx.engine.create_table(&ctx.account_id, input).await.unwrap();

    // Enable TTL
    ctx.engine
        .update_ttl(&ctx.account_id, &table_name, "expires_at", true)
        .await
        .expect("update_ttl failed");

    // Describe TTL
    let ttl_desc = ctx
        .engine
        .describe_ttl(&ctx.account_id, &table_name)
        .await
        .expect("describe_ttl failed");
    assert_eq!(ttl_desc.time_to_live_status, TimeToLiveStatus::Enabled);
    assert_eq!(ttl_desc.attribute_name, Some("expires_at".to_string()));

    let key_info = ctx.engine.table_key_info(&ctx.account_id, &table_name).await.unwrap();
    let empty_maps = ExpressionMaps::default();

    // Ensure TTL index table before putting TTL items
    ensure_ttl_index_table(&ctx.client).await.expect("ensure TTL table");

    // Put expired item (past timestamp)
    let mut expired_item: Item = BTreeMap::new();
    expired_item.insert("session_id".to_string(), AttributeValue::S("sess#expired".to_string()));
    expired_item.insert("expires_at".to_string(), AttributeValue::N("1000".to_string()));
    expired_item.insert("data".to_string(), AttributeValue::S("temp_token".to_string()));
    ctx.engine
        .put_item(&key_info, expired_item, false, None, &empty_maps, None)
        .await
        .unwrap();

    // Put valid item (far future timestamp)
    let mut valid_item: Item = BTreeMap::new();
    valid_item.insert("session_id".to_string(), AttributeValue::S("sess#active".to_string()));
    valid_item.insert("expires_at".to_string(), AttributeValue::N("9999999999".to_string()));
    valid_item.insert("data".to_string(), AttributeValue::S("permanent_token".to_string()));
    ctx.engine
        .put_item(&key_info, valid_item, false, None, &empty_maps, None)
        .await
        .unwrap();

    // Put permanent item without TTL
    let mut perm_item: Item = BTreeMap::new();
    perm_item.insert("session_id".to_string(), AttributeValue::S("sess#permanent".to_string()));
    perm_item.insert("data".to_string(), AttributeValue::S("no_ttl_token".to_string()));
    ctx.engine
        .put_item(&key_info, perm_item, false, None, &empty_maps, None)
        .await
        .unwrap();

    // Execute sweep_once pass
    let sweep_res = sweep_once(&ctx.engine).await;
    assert!(sweep_res.is_ok(), "sweep_once failed: {:?}", sweep_res.err());

    // Verify expired item was removed by sweep
    let mut exp_key: Item = BTreeMap::new();
    exp_key.insert("session_id".to_string(), AttributeValue::S("sess#expired".to_string()));
    let exp_check = ctx.engine.get_item(&key_info, &exp_key).await.unwrap();
    assert!(
        exp_check.is_none(),
        "expired session must be deleted by TTL worker sweep"
    );

    // Verify valid item remains
    let mut act_key: Item = BTreeMap::new();
    act_key.insert("session_id".to_string(), AttributeValue::S("sess#active".to_string()));
    let act_check = ctx.engine.get_item(&key_info, &act_key).await.unwrap();
    assert!(act_check.is_some(), "active session must not be deleted");

    // Verify permanent item without TTL remains
    let mut perm_key: Item = BTreeMap::new();
    perm_key.insert("session_id".to_string(), AttributeValue::S("sess#permanent".to_string()));
    let perm_check = ctx.engine.get_item(&key_info, &perm_key).await.unwrap();
    assert!(perm_check.is_some(), "permanent session must not be deleted");
}
