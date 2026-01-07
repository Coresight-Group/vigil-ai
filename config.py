# VIGIL System Configuration
# Updated for consolidated dualpathtransformer.py

# API CONFIGURATION
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False
FLASK_ENV = 'production'

# DATABASE CONFIGURATION
SUPABASE_URL = 'your_supabase_url'
SUPABASE_KEY = 'your_supabase_key'
SUPABASE_TIMEOUT = 30
VECTOR_DIMENSION = 768

# X.AI GROK CONFIGURATION
XAI_API_KEY = 'your_xai_api_key'
XAI_API_URL = 'https://api.x.ai/v1/chat/completions'
XAI_MODEL = 'grok-2'
XAI_TIMEOUT = 30
XAI_TEMPERATURE_FORMAL = 0.3
XAI_TEMPERATURE_CREATIVE = 0.7
XAI_MAX_TOKENS_STANDARD = 1500

# TRANSFORMER CONFIGURATION
TRANSFORMER_MODEL_NAME = 'dualpathtransformer'
EMBEDDING_MODEL = 'distilbert-base-uncased'
EMBEDDING_DIMENSION = 768
HIDDEN_DIMENSION = 512
NUM_RISK_CATEGORIES = 5
NUM_SEVERITY_LEVELS = 4

# INTEGRATED COMPONENTS (all in dualpathtransformer.py)
TRANSFORMER_COMPONENTS = {
    'grok_intelligence_engine': {'methods': 3},
    'risk_correlation_engine': {'methods': 7},
    'schema_validator': {'methods': 1},
    'document_store': {'methods': 5},
    'dual_path_transformer': {'methods': 6}
}

# INTERCONNECTED ANALYSIS CONFIGURATION
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

# SEARCH PARAMETERS
VECTOR_MATCH_THRESHOLD = 0.7
VECTOR_MATCH_COUNT_DEFAULT = 10
VECTOR_MATCH_COUNT_QUICK = 5

# GOVERNANCE RULES
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

# ALERT CONFIGURATION
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

# SYNTHESIS CONFIGURATION
SYNTHESIS = {
    'show_source_attribution': True,
    'show_confidence_levels': True,
    'show_consensus': True,
    'show_interconnections': True,
    'default_confidence_private': 0.95,
    'default_confidence_vigil': 0.75
}

# FEATURE FLAGS
FEATURES = {
    'dual_source_synthesis': True,
    'alert_generation': True,
    'interconnected_analysis': True,
    'self_conflict_detection': True,
    'historical_correlation': True,
    'pattern_recognition': True,
    'cascade_analysis': True,
    'timeline_correlation': True
}

# PERFORMANCE TUNING
PERFORMANCE = {
    'batch_processing_enabled': True,
    'cache_embeddings': True,
    'async_alerts': True,
    'embedding_cache_ttl': 3600,
    'search_timeout': 30,
    'grok_request_timeout': 30
}

# LOGGING
LOG_LEVEL = 'INFO'
LOG_FILE = 'vigil.log'

# TRANSFORMER INITIALIZATION
TRANSFORMER_INIT = {
    'model_name': EMBEDDING_MODEL,
    'embedding_dim': EMBEDDING_DIMENSION,
    'hidden_dim': HIDDEN_DIMENSION,
    'num_categories': NUM_RISK_CATEGORIES,
    'num_severity_levels': NUM_SEVERITY_LEVELS,
    'grok_api_key': XAI_API_KEY,
    'supabase_client': 'auto'
}
