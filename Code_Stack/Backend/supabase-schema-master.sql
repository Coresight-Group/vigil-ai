-- ============================================================================
-- VIGIL MASTER DATABASE SCHEMA
-- Platform administration and multi-tenant client management
-- Run this ONCE in your master/platform Supabase project
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- MULTI-TENANT CLIENT MANAGEMENT
-- ============================================================================

-- CLIENTS table (registry of all client organizations)
CREATE TABLE IF NOT EXISTS clients (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  slug TEXT UNIQUE NOT NULL,              -- URL-safe identifier (e.g., 'acme-corp')

  -- Supabase project connection details
  supabase_url TEXT NOT NULL,             -- Client's Supabase project URL
  supabase_anon_key TEXT NOT NULL,        -- Client's anon/public key
  supabase_service_key TEXT,              -- Service role key (encrypted, for server-side)
  storage_bucket TEXT DEFAULT 'documents', -- Default storage bucket name

  -- Client settings
  status TEXT DEFAULT 'active',           -- 'active', 'suspended', 'pending', 'archived'
  tier TEXT DEFAULT 'standard',           -- 'free', 'standard', 'professional', 'enterprise'
  max_storage_gb INT DEFAULT 10,
  max_risks INT DEFAULT 10000,
  max_documents INT DEFAULT 1000,

  -- Contact info
  contact_email TEXT,
  contact_name TEXT,
  billing_email TEXT,

  -- Configuration
  settings JSONB DEFAULT '{}',
  -- Example settings:
  -- {
  --   "default_risk_type": "SUPPLY_CHAIN",
  --   "alert_thresholds": {"critical": 0.9, "high": 0.75},
  --   "enabled_features": ["grok_analysis", "document_processing"],
  --   "timezone": "America/New_York"
  -- }

  -- Audit
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  created_by TEXT,
  last_accessed_at TIMESTAMP WITH TIME ZONE
);

-- ============================================================================
-- DATA SOURCES (tracks all connected data sources per client)
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_sources (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,

  -- Source identification
  name TEXT NOT NULL,
  source_type TEXT NOT NULL,              -- 'file_upload', 'api', 'database', 'webhook', 'sftp'

  -- Connection configuration (encrypted sensitive fields)
  connection_config JSONB NOT NULL DEFAULT '{}',
  -- Example for API:
  -- {
  --   "base_url": "https://api.erp-system.com",
  --   "auth_type": "oauth2",
  --   "endpoints": [{"path": "/suppliers", "method": "GET", "map_to": "suppliers"}]
  -- }
  --
  -- Example for Database:
  -- {
  --   "db_type": "postgresql",
  --   "host": "db.client.com",
  --   "port": 5432,
  --   "database": "erp_prod",
  --   "ssl": true
  -- }

  -- Credentials (stored separately, encrypted)
  credentials_encrypted TEXT,

  -- Sync settings
  sync_enabled BOOLEAN DEFAULT TRUE,
  sync_frequency TEXT DEFAULT 'daily',    -- 'realtime', 'hourly', 'daily', 'weekly', 'manual'
  sync_schedule TEXT,                     -- Cron expression for custom schedules
  last_sync_at TIMESTAMP WITH TIME ZONE,
  next_sync_at TIMESTAMP WITH TIME ZONE,

  -- Status tracking
  status TEXT DEFAULT 'pending',          -- 'pending', 'active', 'error', 'disabled', 'syncing'
  health_status TEXT DEFAULT 'unknown',   -- 'healthy', 'degraded', 'unhealthy', 'unknown'
  last_health_check TIMESTAMP WITH TIME ZONE,
  error_message TEXT,
  error_count INT DEFAULT 0,

  -- Field mapping configuration
  field_mappings JSONB DEFAULT '{}',

  -- Statistics
  total_records_synced BIGINT DEFAULT 0,
  last_record_count INT,

  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- SYNC JOBS (ETL job execution tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS sync_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
  data_source_id UUID NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,

  -- Job details
  job_type TEXT NOT NULL,                 -- 'full_sync', 'incremental', 'manual', 'retry'
  status TEXT DEFAULT 'pending',          -- 'pending', 'running', 'completed', 'failed', 'cancelled'

  -- Timing
  started_at TIMESTAMP WITH TIME ZONE,
  completed_at TIMESTAMP WITH TIME ZONE,
  duration_ms INT,

  -- Results
  records_processed INT DEFAULT 0,
  records_created INT DEFAULT 0,
  records_updated INT DEFAULT 0,
  records_failed INT DEFAULT 0,
  records_skipped INT DEFAULT 0,

  -- Error tracking
  error_message TEXT,
  error_details JSONB DEFAULT '[]',

  -- Progress tracking
  progress_percent INT DEFAULT 0,
  current_step TEXT,

  -- Logs
  logs JSONB DEFAULT '[]',

  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- API KEYS (client API credentials for your system)
-- ============================================================================

CREATE TABLE IF NOT EXISTS api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,

  -- Key details
  name TEXT NOT NULL,                     -- Friendly name (e.g., 'Production Key')
  key_hash TEXT NOT NULL,                 -- Hashed API key (never store plaintext)
  key_prefix TEXT NOT NULL,               -- First 8 chars for identification (e.g., 'vgl_prod')

  -- Permissions
  scopes TEXT[] DEFAULT '{}',             -- ['read:risks', 'write:risks', 'read:documents', ...]

  -- Usage limits
  rate_limit_per_minute INT DEFAULT 60,
  rate_limit_per_day INT DEFAULT 10000,

  -- Status
  status TEXT DEFAULT 'active',           -- 'active', 'revoked', 'expired'
  expires_at TIMESTAMP WITH TIME ZONE,
  last_used_at TIMESTAMP WITH TIME ZONE,
  usage_count BIGINT DEFAULT 0,

  -- Audit
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  created_by TEXT,
  revoked_at TIMESTAMP WITH TIME ZONE,
  revoked_by TEXT,
  revoke_reason TEXT
);

-- ============================================================================
-- WEBHOOKS (outbound event notifications)
-- ============================================================================

CREATE TABLE IF NOT EXISTS webhooks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,

  -- Webhook configuration
  name TEXT NOT NULL,
  url TEXT NOT NULL,
  secret TEXT,                            -- For HMAC signature verification

  -- Events to trigger on
  events TEXT[] DEFAULT '{}',             -- ['risk.created', 'risk.critical', 'sync.completed', ...]

  -- Delivery settings
  status TEXT DEFAULT 'active',           -- 'active', 'disabled', 'failing'
  retry_count INT DEFAULT 3,
  timeout_seconds INT DEFAULT 30,

  -- Statistics
  total_deliveries BIGINT DEFAULT 0,
  successful_deliveries BIGINT DEFAULT 0,
  failed_deliveries BIGINT DEFAULT 0,
  last_delivery_at TIMESTAMP WITH TIME ZONE,
  last_delivery_status TEXT,
  last_failure_reason TEXT,
  consecutive_failures INT DEFAULT 0,

  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- USAGE METRICS (track client usage for billing/limits)
-- ============================================================================

CREATE TABLE IF NOT EXISTS usage_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,

  -- Period
  period_start DATE NOT NULL,
  period_end DATE NOT NULL,

  -- Counts
  risks_created INT DEFAULT 0,
  risks_analyzed INT DEFAULT 0,
  documents_uploaded INT DEFAULT 0,
  documents_processed INT DEFAULT 0,
  api_calls INT DEFAULT 0,
  sync_jobs_run INT DEFAULT 0,
  storage_bytes_used BIGINT DEFAULT 0,

  -- AI usage
  grok_tokens_used BIGINT DEFAULT 0,
  embeddings_generated INT DEFAULT 0,

  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

  UNIQUE(client_id, period_start, period_end)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_clients_slug ON clients (slug);
CREATE INDEX IF NOT EXISTS idx_clients_status ON clients (status);

CREATE INDEX IF NOT EXISTS idx_data_sources_client_id ON data_sources (client_id);
CREATE INDEX IF NOT EXISTS idx_data_sources_source_type ON data_sources (source_type);
CREATE INDEX IF NOT EXISTS idx_data_sources_status ON data_sources (status);
CREATE INDEX IF NOT EXISTS idx_data_sources_next_sync ON data_sources (next_sync_at) WHERE sync_enabled = TRUE;

CREATE INDEX IF NOT EXISTS idx_sync_jobs_client_id ON sync_jobs (client_id);
CREATE INDEX IF NOT EXISTS idx_sync_jobs_data_source_id ON sync_jobs (data_source_id);
CREATE INDEX IF NOT EXISTS idx_sync_jobs_status ON sync_jobs (status);
CREATE INDEX IF NOT EXISTS idx_sync_jobs_created_at ON sync_jobs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_api_keys_client_id ON api_keys (client_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_prefix ON api_keys (key_prefix);
CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys (status) WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_webhooks_client_id ON webhooks (client_id);
CREATE INDEX IF NOT EXISTS idx_webhooks_status ON webhooks (status);

CREATE INDEX IF NOT EXISTS idx_usage_metrics_client_id ON usage_metrics (client_id);
CREATE INDEX IF NOT EXISTS idx_usage_metrics_period ON usage_metrics (period_start, period_end);

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE clients ENABLE ROW LEVEL SECURITY;
ALTER TABLE data_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE sync_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE webhooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_metrics ENABLE ROW LEVEL SECURITY;

-- Admin full access policies (your internal system uses service role)
CREATE POLICY "Service role full access clients" ON clients FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access data_sources" ON data_sources FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access sync_jobs" ON sync_jobs FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access api_keys" ON api_keys FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access webhooks" ON webhooks FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access usage_metrics" ON usage_metrics FOR ALL USING (true) WITH CHECK (true);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_clients_timestamp ON clients;
CREATE TRIGGER update_clients_timestamp BEFORE UPDATE ON clients
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_data_sources_timestamp ON data_sources;
CREATE TRIGGER update_data_sources_timestamp BEFORE UPDATE ON data_sources
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_webhooks_timestamp ON webhooks;
CREATE TRIGGER update_webhooks_timestamp BEFORE UPDATE ON webhooks
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Get client by API key prefix
CREATE OR REPLACE FUNCTION get_client_by_api_key_prefix(key_prefix_input TEXT)
RETURNS TABLE(
  client_id UUID,
  client_name TEXT,
  supabase_url TEXT,
  supabase_anon_key TEXT,
  scopes TEXT[]
)
LANGUAGE sql STABLE
AS $$
  SELECT
    c.id as client_id,
    c.name as client_name,
    c.supabase_url,
    c.supabase_anon_key,
    ak.scopes
  FROM api_keys ak
  JOIN clients c ON c.id = ak.client_id
  WHERE ak.key_prefix = key_prefix_input
    AND ak.status = 'active'
    AND c.status = 'active'
    AND (ak.expires_at IS NULL OR ak.expires_at > CURRENT_TIMESTAMP);
$$;

-- Get data sources due for sync
CREATE OR REPLACE FUNCTION get_pending_syncs()
RETURNS TABLE(
  data_source_id UUID,
  client_id UUID,
  source_type TEXT,
  connection_config JSONB,
  last_sync_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE sql STABLE
AS $$
  SELECT
    ds.id as data_source_id,
    ds.client_id,
    ds.source_type,
    ds.connection_config,
    ds.last_sync_at
  FROM data_sources ds
  JOIN clients c ON c.id = ds.client_id
  WHERE ds.sync_enabled = TRUE
    AND ds.status = 'active'
    AND c.status = 'active'
    AND (ds.next_sync_at IS NULL OR ds.next_sync_at <= CURRENT_TIMESTAMP)
  ORDER BY ds.next_sync_at ASC NULLS FIRST;
$$;

-- Increment usage metrics
CREATE OR REPLACE FUNCTION increment_usage_metrics(
  p_client_id UUID,
  p_period_start DATE,
  p_period_end DATE,
  p_risks_created INT DEFAULT 0,
  p_risks_analyzed INT DEFAULT 0,
  p_documents_uploaded INT DEFAULT 0,
  p_documents_processed INT DEFAULT 0,
  p_api_calls INT DEFAULT 0,
  p_grok_tokens BIGINT DEFAULT 0,
  p_embeddings INT DEFAULT 0
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
  INSERT INTO usage_metrics (
    client_id, period_start, period_end,
    risks_created, risks_analyzed, documents_uploaded,
    documents_processed, api_calls, grok_tokens_used, embeddings_generated
  ) VALUES (
    p_client_id, p_period_start, p_period_end,
    p_risks_created, p_risks_analyzed, p_documents_uploaded,
    p_documents_processed, p_api_calls, p_grok_tokens, p_embeddings
  )
  ON CONFLICT (client_id, period_start, period_end) DO UPDATE SET
    risks_created = usage_metrics.risks_created + EXCLUDED.risks_created,
    risks_analyzed = usage_metrics.risks_analyzed + EXCLUDED.risks_analyzed,
    documents_uploaded = usage_metrics.documents_uploaded + EXCLUDED.documents_uploaded,
    documents_processed = usage_metrics.documents_processed + EXCLUDED.documents_processed,
    api_calls = usage_metrics.api_calls + EXCLUDED.api_calls,
    grok_tokens_used = usage_metrics.grok_tokens_used + EXCLUDED.grok_tokens_used,
    embeddings_generated = usage_metrics.embeddings_generated + EXCLUDED.embeddings_generated;
END;
$$;

-- ============================================================================
-- AUTHENTICATION: ADMINS (one per client organization)
-- ============================================================================

CREATE TABLE IF NOT EXISTS admins (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,

  -- Credentials
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,              -- PBKDF2 with salt

  -- Profile
  full_name TEXT NOT NULL,
  phone TEXT,

  -- Status
  status TEXT DEFAULT 'pending',            -- 'pending', 'active', 'suspended', 'locked'
  email_verified BOOLEAN DEFAULT FALSE,

  -- Security
  failed_login_attempts INT DEFAULT 0,
  locked_until TIMESTAMP WITH TIME ZONE,
  last_login_at TIMESTAMP WITH TIME ZONE,
  last_login_ip TEXT,
  password_changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

  -- Audit
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

  UNIQUE(client_id)                         -- One admin per client
);

-- ============================================================================
-- AUTHENTICATION: USERS (multiple per client organization)
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
  created_by UUID REFERENCES admins(id),

  -- Credentials
  email TEXT NOT NULL,
  password_hash TEXT NOT NULL,

  -- Profile
  full_name TEXT NOT NULL,
  department TEXT,
  role TEXT DEFAULT 'analyst',              -- 'analyst', 'manager', 'viewer'

  -- Status
  status TEXT DEFAULT 'active',             -- 'active', 'suspended', 'locked'
  email_verified BOOLEAN DEFAULT FALSE,

  -- Security
  failed_login_attempts INT DEFAULT 0,
  locked_until TIMESTAMP WITH TIME ZONE,
  last_login_at TIMESTAMP WITH TIME ZONE,
  last_login_ip TEXT,

  -- Permissions
  permissions JSONB DEFAULT '{}',
  -- Example: {"can_create_risks": true, "can_delete": false, "max_severity": "HIGH"}

  -- Audit
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

  UNIQUE(client_id, email)                  -- Email unique per client
);

-- ============================================================================
-- SESSIONS (active login sessions)
-- ============================================================================

CREATE TABLE IF NOT EXISTS sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,

  -- Session owner (either admin or user)
  admin_id UUID REFERENCES admins(id) ON DELETE CASCADE,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,

  -- Session details
  token_hash TEXT UNIQUE NOT NULL,          -- Hashed session token
  user_agent TEXT,
  ip_address TEXT,

  -- Timing
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
  last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

  -- Status
  is_active BOOLEAN DEFAULT TRUE,
  revoked_at TIMESTAMP WITH TIME ZONE,
  revoke_reason TEXT,

  -- Ensure session belongs to either admin or user, not both
  CONSTRAINT session_owner_check CHECK (
    (admin_id IS NOT NULL AND user_id IS NULL) OR
    (admin_id IS NULL AND user_id IS NOT NULL)
  )
);

-- ============================================================================
-- AUTH CODES (email verification, password reset)
-- ============================================================================

CREATE TABLE IF NOT EXISTS auth_codes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,

  -- Target (either admin or user)
  admin_id UUID REFERENCES admins(id) ON DELETE CASCADE,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,

  -- Code details
  code TEXT NOT NULL,                       -- 6-digit code
  code_type TEXT NOT NULL,                  -- 'email_verification', 'password_reset', 'login_2fa'

  -- Timing
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
  used_at TIMESTAMP WITH TIME ZONE,

  -- Attempts
  attempts INT DEFAULT 0,
  max_attempts INT DEFAULT 3
);

-- ============================================================================
-- ACTIVITY LOGS (audit trail)
-- ============================================================================

CREATE TABLE IF NOT EXISTS activity_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,

  -- Actor
  admin_id UUID REFERENCES admins(id) ON DELETE SET NULL,
  user_id UUID REFERENCES users(id) ON DELETE SET NULL,

  -- Action details
  action TEXT NOT NULL,                     -- 'login', 'logout', 'create_user', 'analyze_risk', etc.
  resource_type TEXT,                       -- 'risk', 'user', 'document', etc.
  resource_id UUID,

  -- Context
  ip_address TEXT,
  user_agent TEXT,
  details JSONB DEFAULT '{}',

  -- Result
  success BOOLEAN DEFAULT TRUE,
  error_message TEXT,

  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- AUTHENTICATION INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_admins_client_id ON admins (client_id);
CREATE INDEX IF NOT EXISTS idx_admins_email ON admins (email);
CREATE INDEX IF NOT EXISTS idx_admins_status ON admins (status);

CREATE INDEX IF NOT EXISTS idx_users_client_id ON users (client_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users (client_id, email);
CREATE INDEX IF NOT EXISTS idx_users_status ON users (status);

CREATE INDEX IF NOT EXISTS idx_sessions_client_id ON sessions (client_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions (token_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions (expires_at) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_auth_codes_code ON auth_codes (code, code_type);
CREATE INDEX IF NOT EXISTS idx_auth_codes_expires ON auth_codes (expires_at);

CREATE INDEX IF NOT EXISTS idx_activity_logs_client_id ON activity_logs (client_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_created_at ON activity_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_logs_action ON activity_logs (action);

-- ============================================================================
-- AUTHENTICATION RLS
-- ============================================================================

ALTER TABLE admins ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE auth_codes ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access admins" ON admins FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access users" ON users FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access sessions" ON sessions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access auth_codes" ON auth_codes FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access activity_logs" ON activity_logs FOR ALL USING (true) WITH CHECK (true);

-- ============================================================================
-- AUTHENTICATION TRIGGERS
-- ============================================================================

DROP TRIGGER IF EXISTS update_admins_timestamp ON admins;
CREATE TRIGGER update_admins_timestamp BEFORE UPDATE ON admins
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_users_timestamp ON users;
CREATE TRIGGER update_users_timestamp BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

-- ============================================================================
-- AUTHENTICATION HELPER FUNCTIONS
-- ============================================================================

-- Validate login credentials (returns user/admin info if valid)
CREATE OR REPLACE FUNCTION validate_login(
  p_email TEXT,
  p_password_hash TEXT,
  p_is_admin BOOLEAN DEFAULT FALSE
)
RETURNS TABLE(
  account_id UUID,
  client_id UUID,
  full_name TEXT,
  status TEXT,
  account_type TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
  IF p_is_admin THEN
    RETURN QUERY
    SELECT
      a.id as account_id,
      a.client_id,
      a.full_name,
      a.status,
      'admin'::TEXT as account_type
    FROM admins a
    WHERE a.email = p_email
      AND a.password_hash = p_password_hash
      AND a.status = 'active';
  ELSE
    RETURN QUERY
    SELECT
      u.id as account_id,
      u.client_id,
      u.full_name,
      u.status,
      'user'::TEXT as account_type
    FROM users u
    WHERE u.email = p_email
      AND u.password_hash = p_password_hash
      AND u.status = 'active';
  END IF;
END;
$$;

-- Create a new session
CREATE OR REPLACE FUNCTION create_session(
  p_client_id UUID,
  p_admin_id UUID DEFAULT NULL,
  p_user_id UUID DEFAULT NULL,
  p_token_hash TEXT DEFAULT NULL,
  p_user_agent TEXT DEFAULT NULL,
  p_ip_address TEXT DEFAULT NULL,
  p_duration_hours INT DEFAULT 24
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
  v_session_id UUID;
BEGIN
  INSERT INTO sessions (
    client_id, admin_id, user_id, token_hash,
    user_agent, ip_address, expires_at
  ) VALUES (
    p_client_id, p_admin_id, p_user_id,
    COALESCE(p_token_hash, encode(gen_random_bytes(32), 'hex')),
    p_user_agent, p_ip_address,
    CURRENT_TIMESTAMP + (p_duration_hours || ' hours')::INTERVAL
  )
  RETURNING id INTO v_session_id;

  RETURN v_session_id;
END;
$$;

-- Clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INT
LANGUAGE plpgsql
AS $$
DECLARE
  v_count INT;
BEGIN
  DELETE FROM sessions
  WHERE expires_at < CURRENT_TIMESTAMP
    OR is_active = FALSE;

  GET DIAGNOSTICS v_count = ROW_COUNT;
  RETURN v_count;
END;
$$;

-- Log activity
CREATE OR REPLACE FUNCTION log_activity(
  p_client_id UUID,
  p_action TEXT,
  p_admin_id UUID DEFAULT NULL,
  p_user_id UUID DEFAULT NULL,
  p_resource_type TEXT DEFAULT NULL,
  p_resource_id UUID DEFAULT NULL,
  p_ip_address TEXT DEFAULT NULL,
  p_user_agent TEXT DEFAULT NULL,
  p_details JSONB DEFAULT '{}'::JSONB,
  p_success BOOLEAN DEFAULT TRUE,
  p_error_message TEXT DEFAULT NULL
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
  v_log_id UUID;
BEGIN
  INSERT INTO activity_logs (
    client_id, admin_id, user_id, action,
    resource_type, resource_id, ip_address, user_agent,
    details, success, error_message
  ) VALUES (
    p_client_id, p_admin_id, p_user_id, p_action,
    p_resource_type, p_resource_id, p_ip_address, p_user_agent,
    p_details, p_success, p_error_message
  )
  RETURNING id INTO v_log_id;

  RETURN v_log_id;
END;
$$;

-- ============================================================================
-- SAMPLE: Register a new client with admin
-- ============================================================================

-- 1. Create the client
-- INSERT INTO clients (name, slug, supabase_url, supabase_anon_key, contact_email)
-- VALUES (
--   'Acme Corporation',
--   'acme-corp',
--   'https://acme-project-id.supabase.co',
--   'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
--   'admin@acme.com'
-- ) RETURNING id;

-- 2. Create the admin for that client
-- INSERT INTO admins (client_id, email, password_hash, full_name)
-- VALUES (
--   'client-uuid-here',
--   'admin@acme.com',
--   'hashed-password-here',
--   'John Admin'
-- );
