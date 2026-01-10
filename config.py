# VIGIL System Configuration
# Updated for consolidated dualpathtransformer.py with Three-Schema Strategy Support

# =========================================================================
# API CONFIGURATION
# =========================================================================
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False
FLASK_ENV = 'production'

# =========================================================================
# DATABASE CONFIGURATION
# =========================================================================
SUPABASE_URL = 'your_supabase_url'
SUPABASE_KEY = 'your_supabase_key'
SUPABASE_TIMEOUT = 30
VECTOR_DIMENSION = 768

# =========================================================================
# X.AI GROK CONFIGURATION
# =========================================================================
XAI_API_KEY = 'your_xai_api_key'
XAI_API_URL = 'https://api.x.ai/v1/chat/completions'
XAI_MODEL = 'grok-2'
XAI_TIMEOUT = 30
XAI_TEMPERATURE_FORMAL = 0.3
XAI_TEMPERATURE_CREATIVE = 0.7
XAI_MAX_TOKENS_STANDARD = 1500

# =========================================================================
# TRANSFORMER CONFIGURATION
# =========================================================================
TRANSFORMER_MODEL_NAME = 'dualpathtransformer'
EMBEDDING_MODEL = 'distilbert-base-uncased'
EMBEDDING_DIMENSION = 768
HIDDEN_DIMENSION = 512
NUM_RISK_CATEGORIES = 5
NUM_SEVERITY_LEVELS = 4

# =========================================================================
# INTEGRATED COMPONENTS (all in dualpathtransformer.py)
# =========================================================================
TRANSFORMER_COMPONENTS = {
    'grok_intelligence_engine': {'methods': 3},
    'risk_correlation_engine': {'methods': 7},
    'schema_validator': {'methods': 1},
    'document_store': {'methods': 5},
    'dual_path_transformer': {'methods': 6}
}

# =========================================================================
# SCHEMA STRATEGY CONFIGURATION (NEW)
# Defines behavior for Structured, Semi-Structured, Unstructured data
# =========================================================================
SCHEMA_STRATEGIES = {
    'structured': {
        'name': 'Schema on Write',
        'validator': 'StructuredValidator',
        'strategy': 'schema-on-write',
        'rejection_on_error': True,
        'required_fields': ['description', 'risk_type', 'severity'],
        'optional_fields': [],
        'valid_enums': {
            'risk_type': ['SUPPLY_CHAIN', 'PRODUCTION', 'QUALITY', 'FINANCIAL', 'REGULATORY', 'MARKET', 'OPERATIONAL'],
            'severity': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        },
        'description': 'Enforces all required fields with strict validation. Rejects invalid data at write time.',
        'storage': 'PostgreSQL table',
        'confidence_baseline': 0.95
    },
    'semi-structured': {
        'name': 'Hybrid Schema',
        'validator': 'SemiStructuredValidator',
        'strategy': 'hybrid',
        'rejection_on_error': False,
        'required_fields': ['description'],
        'optional_fields': ['risk_type', 'severity'],
        'valid_enums': {
            'risk_type': ['SUPPLY_CHAIN', 'PRODUCTION', 'QUALITY', 'FINANCIAL', 'REGULATORY', 'MARKET', 'OPERATIONAL'],
            'severity': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        },
        'description': 'Accepts core fields with flexible extras. Continues on partial errors. Stores structured + JSONB.',
        'storage': 'PostgreSQL (structured) + JSONB (flexible)',
        'confidence_baseline': 0.88
    },
    'unstructured': {
        'name': 'Schema on Read',
        'validator': 'UnstructuredValidator',
        'strategy': 'schema-on-read',
        'rejection_on_error': False,
        'required_fields': [],
        'optional_fields': ['description'],
        'minimum_validation': True,
        'min_length': 20,
        'max_length': 5000,
        'description': 'Accepts any text. Graceful validation. Validates findings at read time.',
        'storage': 'Document store',
        'confidence_baseline': 0.80
    }
}

# =========================================================================
# DATA TYPE DETECTION CONFIGURATION (NEW)
# =========================================================================
DATA_TYPE_DETECTION = {
    'required_fields_for_structured': 3,
    'minimum_fields_for_semi': 1,
    'unstructured_fallback': True,
    'detection_order': ['structured', 'semi-structured', 'unstructured']
}

# =========================================================================
# ALERT GENERATION RULES (NEW)
# Complexity score determines alert count and levels
# =========================================================================
ALERT_RULES = {
    'simple_risk': {
        'complexity_range': (1, 3),
        'alert_count': 0,
        'description': 'Simple risks with routine handling'
    },
    'medium_risk': {
        'complexity_range': (4, 7),
        'alert_count': 1,
        'alert_level': 'HIGH',
        'description': 'Medium risks requiring executive review'
    },
    'critical_risk': {
        'complexity_range': (8, 10),
        'alert_count': 2,
        'alert_levels': ['CRITICAL', 'HIGH'],
        'description': 'Critical risks requiring crisis response'
    }
}

# =========================================================================
# CONFIDENCE THRESHOLDS (NEW)
# Baseline confidence for each data type
# =========================================================================
CONFIDENCE_LEVELS = {
    'structured': 0.95,
    'semi-structured': 0.88,
    'unstructured': 0.80,
    'descriptions': {
        'structured': 'User provided all info (95%+ confidence)',
        'semi-structured': 'User provided some, system inferred rest (85-90%)',
        'unstructured': 'System inferred all (75-85%)'
    }
}

# =========================================================================
# INTERCONNECTED ANALYSIS CONFIGURATION
# =========================================================================
ANALYSIS_FEATURES = {
    'self_conflict_detection': True,
    'historical_correlation': True,
    'recurring_pattern_detection': True,
    'grok_intelligence': True,
    'cascading_effect_analysis': True,
    'timeline_correlation': True,
    'semantic_search': True,
    'narrative_generation': True
}

ANALYSIS_THRESHOLDS = {
    'similarity_threshold': 0.6,
    'recurring_minimum': 3,
    'recurring_high_severity': 4,
    'timeline_window_days': 14,
    'entity_match_min_length': 3
}

# =========================================================================
# SEARCH PARAMETERS
# =========================================================================
VECTOR_MATCH_THRESHOLD = 0.7
VECTOR_MATCH_COUNT_DEFAULT = 10
VECTOR_MATCH_COUNT_QUICK = 5

# =========================================================================
# GOVERNANCE RULES
# =========================================================================
POLICIES = {
    'supplier_concentration': {
        'green': 0.10,
        'yellow': 0.15,
        'orange': 0.25,
        'red': 1.0
    },
    'incident_frequency': {
        'baseline': 1.0,
        'yellow_threshold': 1.6,
        'red_threshold': 2.3
    }
}

# =========================================================================
# ALERT CONFIGURATION
# =========================================================================
ALERT_SETTINGS = {
    'CRITICAL': {
        'channels': ['SMS', 'EMAIL', 'DASHBOARD', 'SLACK', 'PHONE'],
        'escalate_to': ['VP_OPERATIONS', 'CFO', 'BOARD'],
        'escalate_within_hours': 24
    },
    'HIGH': {
        'channels': ['EMAIL', 'DASHBOARD', 'SLACK'],
        'escalate_to': ['VP_OPERATIONS', 'FINANCE'],
        'escalate_within_hours': 48
    }
}

# =========================================================================
# SYNTHESIS CONFIGURATION
# =========================================================================
SYNTHESIS = {
    'show_source_attribution': True,
    'show_confidence_levels': True,
    'show_consensus': True,
    'show_interconnections': True,
    'default_confidence_private': 0.95,
    'default_confidence_vigil': 0.75
}

# =========================================================================
# FEATURE FLAGS
# =========================================================================
FEATURES = {
    'dual_source_synthesis': True,
    'alert_generation': True,
    'interconnected_analysis': True,
    'self_conflict_detection': True,
    'historical_correlation': True,
    'pattern_recognition': True,
    'cascade_analysis': True,
    'timeline_correlation': True,
    'data_type_detection': True,
    'schema_strategies': True
}

# =========================================================================
# PERFORMANCE TUNING
# =========================================================================
PERFORMANCE = {
    'batch_processing_enabled': True,
    'cache_embeddings': True,
    'async_alerts': True,
    'embedding_cache_ttl': 3600,
    'search_timeout': 30,
    'grok_request_timeout': 30
}

# =========================================================================
# LOGGING
# =========================================================================
LOG_LEVEL = 'INFO'
LOG_FILE = 'vigil.log'

# =========================================================================
# TRANSFORMER INITIALIZATION
# =========================================================================
TRANSFORMER_INIT = {
    'model_name': EMBEDDING_MODEL,
    'embedding_dim': EMBEDDING_DIMENSION,
    'hidden_dim': HIDDEN_DIMENSION,
    'num_categories': NUM_RISK_CATEGORIES,
    'num_severity_levels': NUM_SEVERITY_LEVELS,
    'grok_api_key': XAI_API_KEY,
    'supabase_client': 'auto'
}