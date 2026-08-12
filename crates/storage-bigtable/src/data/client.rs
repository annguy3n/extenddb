//! BigTable connection wrapper providing both data and admin clients.

use std::sync::Arc;
use std::time::Duration;

use bigtable_rs::bigtable::{BigTable, BigTableConnection};
use gcp_auth::TokenProvider;

use crate::config::BigtableStorageConfig;

/// Bundles a data-plane connection to BigTable. The admin-plane handle is
/// built lazily by `data::admin` when needed (it uses a different transport
/// crate and we don't want to pay for its setup when only data ops happen).
pub struct BigtableClient {
    pub project_id: String,
    pub instance_id: String,
    pub emulator_host: Option<String>,
    pub credentials_path: Option<String>,
    connection: BigTableConnection,
}

impl BigtableClient {
    pub async fn connect(config: &BigtableStorageConfig) -> Result<Self, String> {
        let channel_size = config.pool_size.max(1) as usize;
        let timeout = Some(Duration::from_secs(30));

        let connection = if let Some(host) = &config.emulator_host {
            BigTableConnection::new_with_emulator(
                host,
                &config.project_id,
                &config.instance_id,
                /* is_read_only */ false,
                channel_size,
                timeout,
            )
            .map_err(|e| format!("bigtable emulator connect: {e}"))?
        } else {
            // Real BigTable uses rustls (via tonic). The crate has both
            // aws-lc-rs and ring providers transitively enabled (bigtable_rs's
            // tls-aws-lc + tonic's tls-native-roots -> ring), so rustls can't
            // auto-pick. Install aws-lc-rs explicitly once; subsequent calls
            // are no-ops.
            let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

            let token_provider: Arc<dyn TokenProvider> = if let Some(cred_path) = &config.credentials_path {
                let sa = gcp_auth::CustomServiceAccount::from_file(cred_path)
                    .map_err(|e| format!("gcp_auth CustomServiceAccount load: {e}"))?;
                Arc::new(sa)
            } else {
                gcp_auth::provider()
                    .await
                    .map_err(|e| format!("gcp_auth provider: {e}"))?
            };

            BigTableConnection::new_with_token_provider(
                &config.project_id,
                &config.instance_id,
                /* is_read_only */ false,
                channel_size,
                timeout,
                token_provider,
            )
            .map_err(|e| format!("bigtable connect: {e}"))?
        };

        Ok(Self {
            project_id: config.project_id.clone(),
            instance_id: config.instance_id.clone(),
            emulator_host: config.emulator_host.clone(),
            credentials_path: config.credentials_path.clone(),
            connection,
        })
    }

    /// Returns a per-request BigTable data client. Cheap to clone.
    pub fn data(&self) -> BigTable {
        self.connection.client()
    }

    /// Convenience: build a fully-qualified BigTable table name from a short
    /// name. Format: `projects/<project>/instances/<instance>/tables/<table>`.
    pub fn full_table_name(&self, short: &str) -> String {
        format!(
            "projects/{}/instances/{}/tables/{}",
            self.project_id, self.instance_id, short
        )
    }

    /// Instance resource name (admin API needs this for CreateTable parent).
    pub fn instance_name(&self) -> String {
        format!("projects/{}/instances/{}", self.project_id, self.instance_id)
    }
}
