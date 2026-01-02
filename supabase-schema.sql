-- ============================================================================
-- VIGIL Enhanced Supabase Schema with JSONB
-- Replaces supabase-schema.sql with structured metadata support
-- ============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- ENHANCED RISKS TABLE
-- Replaces flat alert_level with structured JSONB metadata
-- ============================================================================

CREATE TABLE IF NOT EXISTS risks (
  id BIGSERIAL PRIMARY KEY,
  
  -- Core risk data
  description TEXT NOT NULL,
  embedding vector(768) NOT NULL,
  
  -- ===== ALERT SYSTEM (JSONB - replaces VARCHAR alert_level) =====
  -- Stores: severity, confidence, trigger conditions, timestamps
  has_alert BOOLEAN DEFAULT FALSE,
  alert_metadata JSONB DEFAULT NULL,
  -- Example alert_metadata:
  -- {
  --   "severity": "CRITICAL",
  --   "confidence": 0.87,
  --   "triggers": [
  --     {
  --       "type": "keyword_match",
  --       "matched_keywords": ["delay", "production"],
  --       "score": 0.92
  --     }
  --   ],
  --   "detected_at": "2024-12-30T12:00:00Z",
  --   "triggered_by_model": "distilbert-base-uncased"
  -- }
  
  -- ===== ANALYSIS RESULTS (JSONB - NEW) =====
  -- Stores: X.AI metadata, confidence, themes, influence factors
  analysis_metadata JSONB DEFAULT NULL,
  -- Example analysis_metadata:
  -- {
  --   "xai_model": "grok-2",
  --   "xai_version": "2024-12-30",
  --   "analysis_confidence": 0.85,
  --   "key_themes": ["supply_chain", "delivery", "quality"],
  --   "similar_risks_count": 3,
  --   "similar_risks_influence": 0.65,
  --   "analysis_tokens_used": 245,
  --   "temperature": 0.7,
  --   "analyzed_at": "2024-12-30T12:00:05Z"
  -- }
  
  -- ===== ATTACHMENT TRACKING (JSONB - NEW) =====
  -- Stores: file metadata, processing status, extracted entities
  attachment_metadata JSONB DEFAULT NULL,
  -- Example attachment_metadata:
  -- {
  --   "filename": "risk_report.pdf",
  --   "file_type": "document",
  --   "file_size": 245600,
  --   "mime_type": "application/pdf",
  --   "uploaded_at": "2024-12-30T12:00:00Z",
  --   "processing_status": "completed",
  --   "extracted_text_length": 1245,
  --   "processing_time_ms": 1200,
  --   "extracted_entities": {
  --     "named_entities": ["Q4", "production"],
  --     "risk_keywords": ["delay", "disruption"]
  --   }
  -- }
  
  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Vector similarity search
CREATE INDEX IF NOT EXISTS idx_risks_embedding 
  ON risks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- JSONB GIN indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_risks_alert_metadata 
  ON risks USING GIN (alert_metadata);

CREATE INDEX IF NOT EXISTS idx_risks_analysis_metadata 
  ON risks USING GIN (analysis_metadata);

CREATE INDEX IF NOT EXISTS idx_risks_attachment_metadata 
  ON risks USING GIN (attachment_metadata);

-- Functional indexes for common queries
CREATE INDEX IF NOT EXISTS idx_risks_alert_severity 
  ON risks ((alert_metadata->>'severity')) 
  WHERE has_alert = true;

CREATE INDEX IF NOT EXISTS idx_risks_analysis_confidence 
  ON risks ((analysis_metadata->>'analysis_confidence')::float);

-- Timestamp indexes
CREATE INDEX IF NOT EXISTS idx_risks_created_at 
  ON risks (created_at DESC);

-- ============================================================================
-- MATCH_RISKS FUNCTION (Vector Similarity Search)
-- ============================================================================

CREATE OR REPLACE FUNCTION match_risks(
  query_embedding vector(768),
  match_threshold float DEFAULT 0.7,
  match_count int DEFAULT 5
)
RETURNS TABLE(
  id BIGINT,
  description TEXT,
  embedding vector(768),
  has_alert BOOLEAN,
  alert_metadata JSONB,
  analysis_metadata JSONB,
  attachment_metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    risks.id,
    risks.description,
    risks.embedding,
    risks.has_alert,
    risks.alert_metadata,
    risks.analysis_metadata,
    risks.attachment_metadata,
    risks.created_at,
    1 - (risks.embedding <=> query_embedding) as similarity
  FROM risks
  WHERE 1 - (risks.embedding <=> query_embedding) > match_threshold
  ORDER BY risks.embedding <=> query_embedding
  LIMIT match_count;
$$;

-- ============================================================================
-- HELPER FUNCTIONS FOR JSONB QUERIES
-- ============================================================================

-- Get risks with critical alerts
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

-- Get high-confidence analyses
CREATE OR REPLACE FUNCTION get_high_confidence_analyses(min_confidence float DEFAULT 0.8)
RETURNS TABLE(
  id BIGINT,
  description TEXT,
  xai_model TEXT,
  analysis_confidence NUMERIC,
  key_themes TEXT,
  created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE sql STABLE
AS $$
  SELECT
    risks.id,
    risks.description,
    analysis_metadata->>'xai_model' as xai_model,
    (analysis_metadata->>'analysis_confidence')::numeric as analysis_confidence,
    analysis_metadata->>'key_themes' as key_themes,
    risks.created_at
  FROM risks
  WHERE analysis_metadata IS NOT NULL
    AND (analysis_metadata->>'analysis_confidence')::float >= min_confidence
  ORDER BY created_at DESC;
$$;

-- Get risks by extracted keywords
CREATE OR REPLACE FUNCTION get_risks_by_keyword(keyword TEXT)
RETURNS TABLE(
  id BIGINT,
  description TEXT,
  file_name TEXT,
  extracted_text_length INT,
  created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE sql STABLE
AS $$
  SELECT
    risks.id,
    risks.description,
    attachment_metadata->>'filename' as file_name,
    (attachment_metadata->>'extracted_text_length')::int as extracted_text_length,
    risks.created_at
  FROM risks
  WHERE attachment_metadata IS NOT NULL
    AND (
      attachment_metadata->'extracted_entities'->'risk_keywords' @> to_jsonb(keyword)::jsonb
      OR attachment_metadata->'extracted_entities'->'named_entities' @> to_jsonb(keyword)::jsonb
    )
  ORDER BY created_at DESC;
$$;

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

ALTER TABLE risks ENABLE ROW LEVEL SECURITY;

-- Allow public read
CREATE POLICY "Allow public read" ON risks
  FOR SELECT
  USING (true);

-- Allow authenticated insert
CREATE POLICY "Allow authenticated insert" ON risks
  FOR INSERT
  WITH CHECK (true);

-- Allow authenticated update
CREATE POLICY "Allow authenticated update" ON risks
  FOR UPDATE
  USING (true)
  WITH CHECK (true);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_risks_updated_at ON risks;
CREATE TRIGGER update_risks_updated_at BEFORE UPDATE ON risks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- MIGRATION HELPER: Convert old alert_level to alert_metadata
-- ============================================================================
-- This will only work if you have existing data with alert_level
-- Run this AFTER deploying the enhanced schema

-- CREATE OR REPLACE FUNCTION migrate_alert_level_to_jsonb()
-- RETURNS void AS $$
-- BEGIN
--   UPDATE risks
--   SET alert_metadata = jsonb_build_object(
--     'severity', alert_level,
--     'confidence', 0.7,
--     'detected_at', created_at,
--     'triggered_by_model', 'distilbert-base-uncased'
--   )
--   WHERE alert_level IS NOT NULL AND alert_metadata IS NULL;
-- END;
-- $$ LANGUAGE plpgsql;

-- -- To run the migration:
-- -- SELECT migrate_alert_level_to_jsonb();

-- ============================================================================
-- SAMPLE DATA FOR TESTING
-- ============================================================================

-- Insert test risk with full metadata
-- INSERT INTO risks (
--   description,
--   embedding,
--   has_alert,
--   alert_metadata,
--   analysis_metadata,
--   attachment_metadata
-- ) VALUES (
--   'Supply chain disruption affecting Q4 production timeline',
--   array_fill(0.1, ARRAY[768])::vector(768),
--   true,
--   jsonb_build_object(
--     'severity', 'HIGH',
--     'confidence', 0.87,
--     'triggers', jsonb_build_array(
--       jsonb_build_object(
--         'type', 'keyword_match',
--         'matched_keywords', jsonb_build_array('disruption', 'Q4'),
--         'score', 0.92
--       )
--     ),
--     'detected_at', now()::text,
--     'triggered_by_model', 'distilbert-base-uncased'
--   ),
--   jsonb_build_object(
--     'xai_model', 'grok-2',
--     'analysis_confidence', 0.85,
--     'key_themes', jsonb_build_array('supply_chain', 'delivery'),
--     'analyzed_at', now()::text
--   ),
--   NULL
-- );

-- ============================================================================
-- USEFUL JSONB QUERIES
-- ============================================================================

-- Find all risks with CRITICAL severity
-- SELECT id, description, alert_metadata->>'severity' as severity
-- FROM risks WHERE alert_metadata->>'severity' = 'CRITICAL';

-- Find risks analyzed with Grok-2
-- SELECT id, description, analysis_metadata->>'xai_model' as model
-- FROM risks WHERE analysis_metadata->>'xai_model' = 'grok-2';

-- Find risks with attached files
-- SELECT id, description, attachment_metadata->>'filename' as filename
-- FROM risks WHERE attachment_metadata IS NOT NULL;

-- Get average analysis confidence
-- SELECT 
--   AVG((analysis_metadata->>'analysis_confidence')::float) as avg_confidence
-- FROM risks 
-- WHERE analysis_metadata IS NOT NULL;

-- ============================================================================
