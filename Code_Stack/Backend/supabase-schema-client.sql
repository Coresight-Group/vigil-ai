-- ============================================================================
-- VIGIL CLIENT DATABASE SCHEMA (COMPLETE)
-- Run this in EACH client's Supabase project
-- Contains: risks, documents, suppliers, inventory, data sync, and all operational data
-- ============================================================================

-- Enable pgvector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- RISKS TABLE (Core risk data with vector embeddings)
-- ============================================================================

CREATE TABLE IF NOT EXISTS risks (
  id BIGSERIAL PRIMARY KEY,

  -- Core risk data
  description TEXT NOT NULL,
  embedding vector(768) NOT NULL,

  -- Risk classification
  risk_type TEXT DEFAULT 'SUPPLY_CHAIN',  -- 'SUPPLY_CHAIN', 'QUALITY', 'DELIVERY', 'PRODUCTION', 'BRAND'
  severity TEXT DEFAULT 'MEDIUM',          -- 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
  confidence DECIMAL(3,2) DEFAULT 0.5,

  -- Structured data from input
  structured_data JSONB DEFAULT '{}',
  -- Example: {"supplier": "Acme Corp", "product": "Widget A", "location": "Shanghai"}

  -- Data type classification
  data_type TEXT DEFAULT 'unstructured',   -- 'structured', 'semi-structured', 'unstructured', 'unified'

  -- Alert system
  has_alert BOOLEAN DEFAULT FALSE,
  alert_metadata JSONB DEFAULT NULL,
  -- Example: {"severity": "CRITICAL", "confidence": 0.87, "triggers": [...]}

  -- Analysis results
  analysis_metadata JSONB DEFAULT NULL,
  -- Example: {"grok_model": "grok-3", "analysis_confidence": 0.85, "key_themes": [...]}

  -- Attachment tracking
  attachment_metadata JSONB DEFAULT NULL,
  -- Example: {"filename": "report.pdf", "file_type": "document", "processing_status": "completed"}

  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DOCUMENTS TABLE (Uploaded files)
-- ============================================================================

CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- File information
  filename TEXT NOT NULL,
  original_filename TEXT NOT NULL,
  file_type TEXT NOT NULL,                -- 'pdf', 'docx', 'xlsx', 'csv', 'txt', 'json', 'image'
  mime_type TEXT NOT NULL,
  file_size BIGINT NOT NULL,

  -- Storage location
  storage_path TEXT NOT NULL,
  storage_bucket TEXT DEFAULT 'documents',

  -- Document classification
  document_category TEXT NOT NULL,        -- 'supplier', 'quality', 'production', 'delivery', 'brand', 'compliance', 'contract', 'report', 'other'
  risk_types TEXT[] DEFAULT '{}',
  solution_categories TEXT[] DEFAULT '{}',

  -- Extracted content
  extracted_text TEXT,
  text_preview TEXT,                      -- First 500 chars
  page_count INT DEFAULT 1,
  word_count INT DEFAULT 0,

  -- Vector embedding
  embedding vector(768),

  -- Metadata
  metadata JSONB DEFAULT '{}',

  -- Processing status
  processing_status TEXT DEFAULT 'pending',  -- 'pending', 'processing', 'completed', 'failed'
  processing_error TEXT,
  processed_at TIMESTAMP WITH TIME ZONE,

  -- Audit
  uploaded_by TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DOCUMENT CHUNKS (for large document search)
-- ============================================================================

CREATE TABLE IF NOT EXISTS document_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

  -- Chunk information
  chunk_index INT NOT NULL,
  chunk_text TEXT NOT NULL,
  chunk_size INT NOT NULL,

  -- Position in original
  start_page INT,
  end_page INT,
  start_position INT,
  end_position INT,

  -- Vector embedding
  embedding vector(768) NOT NULL,

  -- Extracted entities
  entities JSONB DEFAULT '{}',

  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- SUPPLIERS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS suppliers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  contact_info JSONB DEFAULT '{}',
  category TEXT,                          -- 'raw_materials', 'components', 'logistics'
  rating DECIMAL(3,2) DEFAULT 0,
  lead_time_days INT,
  minimum_order DECIMAL(12,2),
  location TEXT,
  region TEXT,
  certifications TEXT[] DEFAULT '{}',
  risk_score DECIMAL(3,2) DEFAULT 0,
  is_active BOOLEAN DEFAULT TRUE,
  is_preferred BOOLEAN DEFAULT FALSE,
  contract_id UUID,
  notes TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INVENTORY ITEMS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS inventory_items (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  sku TEXT UNIQUE,
  category TEXT,
  quantity INT DEFAULT 0,
  reorder_level INT DEFAULT 0,
  unit_cost DECIMAL(12,2),
  location TEXT,
  supplier_id UUID REFERENCES suppliers(id),
  is_critical BOOLEAN DEFAULT FALSE,
  lead_time_days INT,
  safety_stock INT DEFAULT 0,
  last_restocked TIMESTAMP WITH TIME ZONE,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- CONTRACTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS contracts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  supplier_id UUID REFERENCES suppliers(id),
  contract_type TEXT,
  start_date DATE,
  end_date DATE,
  value DECIMAL(15,2),
  terms JSONB DEFAULT '{}',
  status TEXT DEFAULT 'active',
  renewal_notice_days INT DEFAULT 30,
  document_id UUID REFERENCES documents(id),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- LOGISTICS PROVIDERS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS logistics_providers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  provider_type TEXT,
  service_areas TEXT[] DEFAULT '{}',
  on_time_rate DECIMAL(3,2) DEFAULT 0.85,
  damage_rate DECIMAL(5,4) DEFAULT 0.01,
  cost_per_unit DECIMAL(10,2),
  contact_info JSONB DEFAULT '{}',
  is_active BOOLEAN DEFAULT TRUE,
  is_preferred BOOLEAN DEFAULT FALSE,
  certifications TEXT[] DEFAULT '{}',
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- CARRIERS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS carriers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  carrier_type TEXT,
  service_level TEXT,
  coverage_areas TEXT[] DEFAULT '{}',
  on_time_rate DECIMAL(3,2) DEFAULT 0.90,
  cost_structure JSONB DEFAULT '{}',
  tracking_capability BOOLEAN DEFAULT TRUE,
  is_active BOOLEAN DEFAULT TRUE,
  contact_info JSONB DEFAULT '{}',
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- SHIPPING ROUTES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS shipping_routes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  origin TEXT NOT NULL,
  destination TEXT NOT NULL,
  route_type TEXT,
  carrier_id UUID REFERENCES carriers(id),
  transit_time_days INT,
  cost_estimate DECIMAL(10,2),
  is_express BOOLEAN DEFAULT FALSE,
  success_rate DECIMAL(3,2) DEFAULT 0.95,
  is_active BOOLEAN DEFAULT TRUE,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- PACKAGING OPTIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS packaging_options (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  packaging_type TEXT,
  protection_level TEXT,
  dimensions JSONB DEFAULT '{}',
  weight_capacity DECIMAL(10,2),
  cost_per_unit DECIMAL(10,2),
  suitable_for TEXT[] DEFAULT '{}',
  is_active BOOLEAN DEFAULT TRUE,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- EQUIPMENT TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS equipment (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  equipment_type TEXT,
  manufacturer TEXT,
  model TEXT,
  serial_number TEXT,
  status TEXT DEFAULT 'operational',
  location TEXT,
  purchase_date DATE,
  warranty_end DATE,
  last_maintenance TIMESTAMP WITH TIME ZONE,
  next_maintenance TIMESTAMP WITH TIME ZONE,
  maintenance_interval_days INT,
  criticality TEXT DEFAULT 'medium',
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- SPARE PARTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS spare_parts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  part_number TEXT,
  compatible_equipment TEXT[] DEFAULT '{}',
  quantity INT DEFAULT 0,
  reorder_level INT DEFAULT 5,
  unit_cost DECIMAL(10,2),
  supplier_id UUID REFERENCES suppliers(id),
  lead_time_days INT,
  is_critical BOOLEAN DEFAULT FALSE,
  location TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- MAINTENANCE SCHEDULES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS maintenance_schedules (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  equipment_id UUID REFERENCES equipment(id),
  schedule_type TEXT,
  frequency TEXT,
  last_performed TIMESTAMP WITH TIME ZONE,
  next_due TIMESTAMP WITH TIME ZONE,
  is_overdue BOOLEAN DEFAULT FALSE,
  assigned_technician TEXT,
  estimated_duration_hours DECIMAL(5,2),
  checklist JSONB DEFAULT '[]',
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- QUALITY STANDARDS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS quality_standards (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  standard_type TEXT,
  standard_code TEXT,
  description TEXT,
  requirements JSONB DEFAULT '[]',
  applicable_to TEXT[] DEFAULT '{}',
  compliance_status TEXT DEFAULT 'compliant',
  last_audit DATE,
  next_audit DATE,
  document_id UUID REFERENCES documents(id),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- CERTIFICATIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS certifications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  certification_body TEXT,
  certification_number TEXT,
  issue_date DATE,
  expiry_date DATE,
  status TEXT DEFAULT 'active',
  scope TEXT,
  applicable_products TEXT[] DEFAULT '{}',
  document_id UUID REFERENCES documents(id),
  renewal_lead_days INT DEFAULT 90,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TESTING EQUIPMENT TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS testing_equipment (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  equipment_type TEXT,
  manufacturer TEXT,
  model TEXT,
  calibration_status TEXT DEFAULT 'calibrated',
  last_calibration DATE,
  next_calibration DATE,
  calibration_interval_days INT,
  accuracy_specs JSONB DEFAULT '{}',
  location TEXT,
  is_active BOOLEAN DEFAULT TRUE,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- PR AGENCIES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS pr_agencies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  specialty TEXT,
  contact_info JSONB DEFAULT '{}',
  retainer_status TEXT,
  success_rate DECIMAL(3,2) DEFAULT 0.8,
  past_campaigns JSONB DEFAULT '[]',
  pricing_tier TEXT,
  is_preferred BOOLEAN DEFAULT FALSE,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- COMMUNICATION TEMPLATES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS communication_templates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  template_type TEXT,
  audience TEXT,
  subject_line TEXT,
  body_template TEXT NOT NULL,
  placeholders JSONB DEFAULT '[]',
  approved BOOLEAN DEFAULT FALSE,
  approved_by TEXT,
  approved_at TIMESTAMP WITH TIME ZONE,
  last_used TIMESTAMP WITH TIME ZONE,
  usage_count INT DEFAULT 0,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- STAKEHOLDER CONTACTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS stakeholder_contacts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  role TEXT,
  organization TEXT,
  contact_info JSONB DEFAULT '{}',
  priority TEXT DEFAULT 'standard',
  communication_preferences JSONB DEFAULT '{}',
  last_contacted TIMESTAMP WITH TIME ZONE,
  relationship_owner TEXT,
  notes TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- MEDIA CONTACTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS media_contacts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  outlet TEXT,
  role TEXT,
  beat TEXT,
  contact_info JSONB DEFAULT '{}',
  relationship_status TEXT,
  past_coverage JSONB DEFAULT '[]',
  preferred_pitch_style TEXT,
  is_key_contact BOOLEAN DEFAULT FALSE,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- PROCUREMENT HISTORY TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS procurement_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  risk_id UUID,
  item_id UUID REFERENCES inventory_items(id),
  supplier_id UUID REFERENCES suppliers(id),
  quantity INT,
  status TEXT DEFAULT 'suggested',
  suggested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  approved_at TIMESTAMP WITH TIME ZONE,
  ordered_at TIMESTAMP WITH TIME ZONE,
  delivered_at TIMESTAMP WITH TIME ZONE,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- AUDIT HISTORY TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  audit_type TEXT NOT NULL,
  audit_name TEXT NOT NULL,
  auditor TEXT,
  audit_date DATE NOT NULL,
  status TEXT DEFAULT 'scheduled',
  scope TEXT,
  findings JSONB DEFAULT '[]',
  corrective_actions JSONB DEFAULT '[]',
  compliance_score DECIMAL(5,2),
  next_audit_date DATE,
  related_standard_id UUID REFERENCES quality_standards(id),
  related_certification_id UUID REFERENCES certifications(id),
  document_id UUID REFERENCES documents(id),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- PRODUCTION LINES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS production_lines (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  line_code TEXT UNIQUE,
  facility TEXT,
  line_type TEXT,
  status TEXT DEFAULT 'operational',
  capacity_per_hour INT,
  current_utilization DECIMAL(5,2),
  products TEXT[] DEFAULT '{}',
  equipment_ids UUID[] DEFAULT '{}',
  shift_schedule JSONB DEFAULT '{}',
  efficiency_rating DECIMAL(3,2),
  last_maintenance TIMESTAMP WITH TIME ZONE,
  next_maintenance TIMESTAMP WITH TIME ZONE,
  downtime_hours_ytd DECIMAL(10,2) DEFAULT 0,
  notes TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DATA SYNC: SYNCED RECORDS TRACKING
-- Tracks all data synced from external sources
-- ============================================================================

CREATE TABLE IF NOT EXISTS synced_records (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Source identification
  source_id TEXT NOT NULL,                -- Original ID from external source
  source_type TEXT NOT NULL,              -- 'google_drive', 'api', 'database', 'sftp'
  source_name TEXT,                       -- Human-readable source name

  -- Target table info
  target_table TEXT NOT NULL,             -- Table where data was stored
  target_record_id UUID,                  -- ID of the created/updated record

  -- Sync metadata
  sync_job_id UUID,                       -- Reference to sync_jobs in master DB
  synced_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  sync_status TEXT DEFAULT 'synced',      -- 'synced', 'pending', 'error', 'deleted'

  -- Change tracking
  action TEXT NOT NULL,                   -- 'created', 'updated', 'deleted'
  previous_hash TEXT,                     -- Hash of previous data (for change detection)
  current_hash TEXT,                      -- Hash of current data

  -- Raw data storage (for debugging/auditing)
  raw_data JSONB DEFAULT '{}',

  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

  -- Unique constraint to prevent duplicate syncs
  UNIQUE(source_id, source_type, target_table)
);

-- ============================================================================
-- DATA SYNC: EXTERNAL FILES
-- Stores metadata for files synced from external sources
-- ============================================================================

CREATE TABLE IF NOT EXISTS external_files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Source information
  source_id TEXT NOT NULL,                -- Original ID (e.g., Google Drive file ID)
  source_type TEXT NOT NULL,              -- 'google_drive', 'dropbox', 'sftp', 'sharepoint'
  source_path TEXT,                       -- Original path in source system

  -- File information
  filename TEXT NOT NULL,
  original_filename TEXT NOT NULL,
  file_type TEXT NOT NULL,                -- 'pdf', 'docx', 'xlsx', 'csv', 'txt', 'json', 'image'
  mime_type TEXT NOT NULL,
  file_size BIGINT NOT NULL,

  -- Storage (local copy in Supabase Storage)
  storage_path TEXT,                      -- Path in Supabase Storage bucket
  storage_bucket TEXT DEFAULT 'synced_files',
  has_local_copy BOOLEAN DEFAULT FALSE,

  -- Web link to original
  web_link TEXT,                          -- URL to view in source system
  download_link TEXT,                     -- URL to download from source

  -- Sync status
  sync_status TEXT DEFAULT 'pending',     -- 'pending', 'synced', 'processing', 'error'
  last_synced_at TIMESTAMP WITH TIME ZONE,
  source_modified_at TIMESTAMP WITH TIME ZONE,

  -- Processing status (for document extraction)
  processing_status TEXT DEFAULT 'pending',  -- 'pending', 'processing', 'completed', 'failed'
  extracted_text TEXT,
  text_preview TEXT,                      -- First 500 chars for quick display
  processing_error TEXT,

  -- Vector embedding for semantic search
  embedding vector(768),

  -- Metadata
  metadata JSONB DEFAULT '{}',
  -- Example metadata:
  -- {
  --   "author": "John Smith",
  --   "owner": "user@company.com",
  --   "shared_with": ["team@company.com"],
  --   "tags": ["supplier", "contract"],
  --   "source_folder": "Contracts/2024",
  --   "permissions": "view"
  -- }

  -- Audit
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

  -- Unique constraint
  UNIQUE(source_id, source_type)
);

-- ============================================================================
-- DATA SYNC: EXTERNAL API DATA
-- Stores data fetched from external REST APIs
-- ============================================================================

CREATE TABLE IF NOT EXISTS external_api_data (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Source identification
  source_id TEXT,                         -- Original ID if provided
  source_endpoint TEXT NOT NULL,          -- API endpoint path
  source_name TEXT,                       -- Human-readable API name

  -- Data
  data_type TEXT NOT NULL,                -- Type of data: 'supplier', 'inventory', 'order', etc.
  data JSONB NOT NULL,                    -- The actual data from API

  -- Sync tracking
  synced_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  sync_source TEXT DEFAULT 'api',

  -- Change detection
  data_hash TEXT,                         -- Hash for change detection
  version INT DEFAULT 1,                  -- Version number for tracking updates

  -- Timestamps from source
  source_created_at TIMESTAMP WITH TIME ZONE,
  source_updated_at TIMESTAMP WITH TIME ZONE,

  -- Local timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DATA SYNC: DATABASE SYNC LOG
-- Tracks data synced from external databases
-- ============================================================================

CREATE TABLE IF NOT EXISTS database_sync_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Source database info
  source_db_type TEXT NOT NULL,           -- 'postgresql', 'mysql', 'mssql'
  source_table TEXT NOT NULL,             -- Source table name
  source_record_id TEXT,                  -- Original record ID

  -- Target info
  target_table TEXT NOT NULL,             -- Target table in this database
  target_record_id UUID,                  -- Created/updated record ID

  -- Sync details
  action TEXT NOT NULL,                   -- 'insert', 'update', 'delete'
  changed_fields TEXT[],                  -- Fields that were changed

  -- Data snapshots
  old_data JSONB DEFAULT '{}',
  new_data JSONB DEFAULT '{}',

  -- Timestamps
  source_timestamp TIMESTAMP WITH TIME ZONE,
  synced_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Risk indexes
CREATE INDEX IF NOT EXISTS idx_risks_embedding ON risks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_risks_risk_type ON risks (risk_type);
CREATE INDEX IF NOT EXISTS idx_risks_severity ON risks (severity);
CREATE INDEX IF NOT EXISTS idx_risks_data_type ON risks (data_type);
CREATE INDEX IF NOT EXISTS idx_risks_structured_data ON risks USING GIN (structured_data);
CREATE INDEX IF NOT EXISTS idx_risks_alert_metadata ON risks USING GIN (alert_metadata);
CREATE INDEX IF NOT EXISTS idx_risks_analysis_metadata ON risks USING GIN (analysis_metadata);
CREATE INDEX IF NOT EXISTS idx_risks_created_at ON risks (created_at DESC);

-- Document indexes
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents (document_category);
CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents (file_type);
CREATE INDEX IF NOT EXISTS idx_documents_risk_types ON documents USING GIN (risk_types);
CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON documents (processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN (metadata);

-- Document chunks indexes
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks (document_id);

-- Solution table indexes
CREATE INDEX IF NOT EXISTS idx_suppliers_category ON suppliers (category);
CREATE INDEX IF NOT EXISTS idx_suppliers_is_preferred ON suppliers (is_preferred) WHERE is_preferred = TRUE;
CREATE INDEX IF NOT EXISTS idx_inventory_items_is_critical ON inventory_items (is_critical) WHERE is_critical = TRUE;
CREATE INDEX IF NOT EXISTS idx_inventory_items_quantity ON inventory_items (quantity);
CREATE INDEX IF NOT EXISTS idx_equipment_status ON equipment (status);
CREATE INDEX IF NOT EXISTS idx_spare_parts_is_critical ON spare_parts (is_critical) WHERE is_critical = TRUE;
CREATE INDEX IF NOT EXISTS idx_maintenance_schedules_is_overdue ON maintenance_schedules (is_overdue) WHERE is_overdue = TRUE;
CREATE INDEX IF NOT EXISTS idx_certifications_status ON certifications (status);
CREATE INDEX IF NOT EXISTS idx_certifications_expiry ON certifications (expiry_date);
CREATE INDEX IF NOT EXISTS idx_procurement_history_status ON procurement_history (status);
CREATE INDEX IF NOT EXISTS idx_audit_history_audit_type ON audit_history (audit_type);
CREATE INDEX IF NOT EXISTS idx_audit_history_status ON audit_history (status);
CREATE INDEX IF NOT EXISTS idx_production_lines_status ON production_lines (status);

-- Synced records indexes
CREATE INDEX IF NOT EXISTS idx_synced_records_source ON synced_records (source_id, source_type);
CREATE INDEX IF NOT EXISTS idx_synced_records_target ON synced_records (target_table, target_record_id);
CREATE INDEX IF NOT EXISTS idx_synced_records_synced_at ON synced_records (synced_at DESC);
CREATE INDEX IF NOT EXISTS idx_synced_records_status ON synced_records (sync_status);

-- External files indexes
CREATE INDEX IF NOT EXISTS idx_external_files_source ON external_files (source_id, source_type);
CREATE INDEX IF NOT EXISTS idx_external_files_sync_status ON external_files (sync_status);
CREATE INDEX IF NOT EXISTS idx_external_files_processing ON external_files (processing_status);
CREATE INDEX IF NOT EXISTS idx_external_files_embedding ON external_files USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_external_files_metadata ON external_files USING GIN (metadata);

-- External API data indexes
CREATE INDEX IF NOT EXISTS idx_external_api_data_source ON external_api_data (source_endpoint, source_name);
CREATE INDEX IF NOT EXISTS idx_external_api_data_type ON external_api_data (data_type);
CREATE INDEX IF NOT EXISTS idx_external_api_data_synced ON external_api_data (synced_at DESC);
CREATE INDEX IF NOT EXISTS idx_external_api_data_content ON external_api_data USING GIN (data);

-- Database sync log indexes
CREATE INDEX IF NOT EXISTS idx_db_sync_log_source ON database_sync_log (source_db_type, source_table);
CREATE INDEX IF NOT EXISTS idx_db_sync_log_target ON database_sync_log (target_table, target_record_id);
CREATE INDEX IF NOT EXISTS idx_db_sync_log_synced ON database_sync_log (synced_at DESC);

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE risks ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE suppliers ENABLE ROW LEVEL SECURITY;
ALTER TABLE inventory_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE contracts ENABLE ROW LEVEL SECURITY;
ALTER TABLE logistics_providers ENABLE ROW LEVEL SECURITY;
ALTER TABLE carriers ENABLE ROW LEVEL SECURITY;
ALTER TABLE shipping_routes ENABLE ROW LEVEL SECURITY;
ALTER TABLE packaging_options ENABLE ROW LEVEL SECURITY;
ALTER TABLE equipment ENABLE ROW LEVEL SECURITY;
ALTER TABLE spare_parts ENABLE ROW LEVEL SECURITY;
ALTER TABLE maintenance_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE quality_standards ENABLE ROW LEVEL SECURITY;
ALTER TABLE certifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE testing_equipment ENABLE ROW LEVEL SECURITY;
ALTER TABLE pr_agencies ENABLE ROW LEVEL SECURITY;
ALTER TABLE communication_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE stakeholder_contacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE media_contacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE procurement_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE production_lines ENABLE ROW LEVEL SECURITY;
ALTER TABLE synced_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE external_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE external_api_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE database_sync_log ENABLE ROW LEVEL SECURITY;

-- Allow authenticated read for all tables
CREATE POLICY "Allow authenticated read risks" ON risks FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read documents" ON documents FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read document_chunks" ON document_chunks FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read suppliers" ON suppliers FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read inventory_items" ON inventory_items FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read contracts" ON contracts FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read logistics_providers" ON logistics_providers FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read carriers" ON carriers FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read shipping_routes" ON shipping_routes FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read packaging_options" ON packaging_options FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read equipment" ON equipment FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read spare_parts" ON spare_parts FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read maintenance_schedules" ON maintenance_schedules FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read quality_standards" ON quality_standards FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read certifications" ON certifications FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read testing_equipment" ON testing_equipment FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read pr_agencies" ON pr_agencies FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read communication_templates" ON communication_templates FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read stakeholder_contacts" ON stakeholder_contacts FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read media_contacts" ON media_contacts FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read procurement_history" ON procurement_history FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read audit_history" ON audit_history FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read production_lines" ON production_lines FOR SELECT TO authenticated USING (true);

-- Allow all for sync tables (client DB is per-tenant, so RLS is less critical)
CREATE POLICY "Allow all synced_records" ON synced_records FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all external_files" ON external_files FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all external_api_data" ON external_api_data FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all database_sync_log" ON database_sync_log FOR ALL USING (true) WITH CHECK (true);

-- Allow authenticated insert/update for key tables
CREATE POLICY "Allow authenticated insert risks" ON risks FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "Allow authenticated update risks" ON risks FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Allow authenticated insert documents" ON documents FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "Allow authenticated update documents" ON documents FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Allow authenticated insert document_chunks" ON document_chunks FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "Allow authenticated insert suppliers" ON suppliers FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "Allow authenticated update suppliers" ON suppliers FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Allow authenticated insert procurement_history" ON procurement_history FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "Allow authenticated update procurement_history" ON procurement_history FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

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

-- Apply to all tables with updated_at
DROP TRIGGER IF EXISTS update_risks_timestamp ON risks;
CREATE TRIGGER update_risks_timestamp BEFORE UPDATE ON risks FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_documents_timestamp ON documents;
CREATE TRIGGER update_documents_timestamp BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_suppliers_timestamp ON suppliers;
CREATE TRIGGER update_suppliers_timestamp BEFORE UPDATE ON suppliers FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_inventory_items_timestamp ON inventory_items;
CREATE TRIGGER update_inventory_items_timestamp BEFORE UPDATE ON inventory_items FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_equipment_timestamp ON equipment;
CREATE TRIGGER update_equipment_timestamp BEFORE UPDATE ON equipment FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_maintenance_schedules_timestamp ON maintenance_schedules;
CREATE TRIGGER update_maintenance_schedules_timestamp BEFORE UPDATE ON maintenance_schedules FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_procurement_history_timestamp ON procurement_history;
CREATE TRIGGER update_procurement_history_timestamp BEFORE UPDATE ON procurement_history FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_audit_history_timestamp ON audit_history;
CREATE TRIGGER update_audit_history_timestamp BEFORE UPDATE ON audit_history FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_production_lines_timestamp ON production_lines;
CREATE TRIGGER update_production_lines_timestamp BEFORE UPDATE ON production_lines FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_synced_records_timestamp ON synced_records;
CREATE TRIGGER update_synced_records_timestamp BEFORE UPDATE ON synced_records FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_external_files_timestamp ON external_files;
CREATE TRIGGER update_external_files_timestamp BEFORE UPDATE ON external_files FOR EACH ROW EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_external_api_data_timestamp ON external_api_data;
CREATE TRIGGER update_external_api_data_timestamp BEFORE UPDATE ON external_api_data FOR EACH ROW EXECUTE FUNCTION update_timestamp();

-- ============================================================================
-- VECTOR SEARCH FUNCTIONS
-- ============================================================================

-- Match risks by semantic similarity
CREATE OR REPLACE FUNCTION match_risks(
  query_embedding vector(768),
  match_threshold float DEFAULT 0.7,
  match_count int DEFAULT 5,
  filter_risk_type TEXT DEFAULT NULL,
  filter_severity TEXT DEFAULT NULL
)
RETURNS TABLE(
  id BIGINT,
  description TEXT,
  risk_type TEXT,
  severity TEXT,
  confidence DECIMAL(3,2),
  structured_data JSONB,
  data_type TEXT,
  has_alert BOOLEAN,
  alert_metadata JSONB,
  analysis_metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    risks.id,
    risks.description,
    risks.risk_type,
    risks.severity,
    risks.confidence,
    risks.structured_data,
    risks.data_type,
    risks.has_alert,
    risks.alert_metadata,
    risks.analysis_metadata,
    risks.created_at,
    1 - (risks.embedding <=> query_embedding) as similarity
  FROM risks
  WHERE
    (filter_risk_type IS NULL OR risks.risk_type = filter_risk_type)
    AND (filter_severity IS NULL OR risks.severity = filter_severity)
    AND 1 - (risks.embedding <=> query_embedding) > match_threshold
  ORDER BY risks.embedding <=> query_embedding
  LIMIT match_count;
$$;

-- Match documents by semantic similarity
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(768),
  match_threshold float DEFAULT 0.7,
  match_count int DEFAULT 10,
  filter_category TEXT DEFAULT NULL,
  filter_risk_types TEXT[] DEFAULT NULL
)
RETURNS TABLE(
  id UUID,
  filename TEXT,
  document_category TEXT,
  risk_types TEXT[],
  text_preview TEXT,
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    documents.id,
    documents.filename,
    documents.document_category,
    documents.risk_types,
    documents.text_preview,
    documents.metadata,
    documents.created_at,
    1 - (documents.embedding <=> query_embedding) as similarity
  FROM documents
  WHERE
    documents.processing_status = 'completed'
    AND documents.embedding IS NOT NULL
    AND 1 - (documents.embedding <=> query_embedding) > match_threshold
    AND (filter_category IS NULL OR documents.document_category = filter_category)
    AND (filter_risk_types IS NULL OR documents.risk_types && filter_risk_types)
  ORDER BY documents.embedding <=> query_embedding
  LIMIT match_count;
$$;

-- Match document chunks
CREATE OR REPLACE FUNCTION match_document_chunks(
  query_embedding vector(768),
  match_threshold float DEFAULT 0.7,
  match_count int DEFAULT 20
)
RETURNS TABLE(
  chunk_id UUID,
  document_id UUID,
  chunk_text TEXT,
  chunk_index INT,
  entities JSONB,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    document_chunks.id as chunk_id,
    document_chunks.document_id,
    document_chunks.chunk_text,
    document_chunks.chunk_index,
    document_chunks.entities,
    1 - (document_chunks.embedding <=> query_embedding) as similarity
  FROM document_chunks
  WHERE 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
  ORDER BY document_chunks.embedding <=> query_embedding
  LIMIT match_count;
$$;

-- Search external files by semantic similarity
CREATE OR REPLACE FUNCTION search_external_files(
  query_embedding vector(768),
  match_threshold float DEFAULT 0.7,
  match_count int DEFAULT 10
)
RETURNS TABLE(
  id UUID,
  filename TEXT,
  source_type TEXT,
  text_preview TEXT,
  metadata JSONB,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    external_files.id,
    external_files.filename,
    external_files.source_type,
    external_files.text_preview,
    external_files.metadata,
    1 - (external_files.embedding <=> query_embedding) as similarity
  FROM external_files
  WHERE external_files.embedding IS NOT NULL
    AND external_files.processing_status = 'completed'
    AND 1 - (external_files.embedding <=> query_embedding) > match_threshold
  ORDER BY external_files.embedding <=> query_embedding
  LIMIT match_count;
$$;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Get critical risks
CREATE OR REPLACE FUNCTION get_critical_risks()
RETURNS TABLE(
  id BIGINT,
  description TEXT,
  severity TEXT,
  confidence NUMERIC,
  created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE sql STABLE
AS $$
  SELECT
    risks.id,
    risks.description,
    alert_metadata->>'severity' as severity,
    (alert_metadata->>'confidence')::numeric as confidence,
    risks.created_at
  FROM risks
  WHERE has_alert = true
    AND alert_metadata->>'severity' IN ('CRITICAL', 'HIGH')
  ORDER BY created_at DESC;
$$;

-- Get expiring certifications
CREATE OR REPLACE FUNCTION get_expiring_certifications(days_ahead INT DEFAULT 90)
RETURNS TABLE(
  id UUID,
  name TEXT,
  expiry_date DATE,
  days_until_expiry INT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    id,
    name,
    expiry_date,
    (expiry_date - CURRENT_DATE)::INT as days_until_expiry
  FROM certifications
  WHERE status = 'active'
    AND expiry_date <= CURRENT_DATE + (days_ahead || ' days')::INTERVAL
  ORDER BY expiry_date ASC;
$$;

-- Get low stock items
CREATE OR REPLACE FUNCTION get_low_stock_items()
RETURNS TABLE(
  id UUID,
  name TEXT,
  quantity INT,
  reorder_level INT,
  is_critical BOOLEAN
)
LANGUAGE sql STABLE
AS $$
  SELECT id, name, quantity, reorder_level, is_critical
  FROM inventory_items
  WHERE quantity <= reorder_level
  ORDER BY is_critical DESC, quantity ASC;
$$;

-- Get overdue maintenance
CREATE OR REPLACE FUNCTION get_overdue_maintenance()
RETURNS TABLE(
  id UUID,
  name TEXT,
  equipment_id UUID,
  next_due TIMESTAMP WITH TIME ZONE,
  days_overdue INT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    id,
    name,
    equipment_id,
    next_due,
    EXTRACT(DAY FROM CURRENT_TIMESTAMP - next_due)::INT as days_overdue
  FROM maintenance_schedules
  WHERE next_due < CURRENT_TIMESTAMP
  ORDER BY next_due ASC;
$$;

-- Get files pending processing
CREATE OR REPLACE FUNCTION get_files_pending_processing(limit_count INT DEFAULT 10)
RETURNS TABLE(
  id UUID,
  source_id TEXT,
  source_type TEXT,
  filename TEXT,
  file_type TEXT,
  storage_path TEXT,
  sync_status TEXT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    external_files.id,
    external_files.source_id,
    external_files.source_type,
    external_files.filename,
    external_files.file_type,
    external_files.storage_path,
    external_files.sync_status
  FROM external_files
  WHERE processing_status = 'pending'
    AND has_local_copy = TRUE
  ORDER BY created_at ASC
  LIMIT limit_count;
$$;

-- Get recent sync activity
CREATE OR REPLACE FUNCTION get_recent_sync_activity(hours_back INT DEFAULT 24)
RETURNS TABLE(
  source_type TEXT,
  action TEXT,
  record_count BIGINT,
  last_synced TIMESTAMP WITH TIME ZONE
)
LANGUAGE sql STABLE
AS $$
  SELECT
    source_type,
    action,
    COUNT(*) as record_count,
    MAX(synced_at) as last_synced
  FROM synced_records
  WHERE synced_at > NOW() - (hours_back || ' hours')::INTERVAL
  GROUP BY source_type, action
  ORDER BY last_synced DESC;
$$;
