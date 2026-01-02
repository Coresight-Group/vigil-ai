# VIGIL System Configuration
# Enhanced configuration with dual-source synthesis, alerts, and all components

# ═══════════════════════════════════════════════════════════════
# API CONFIGURATION
# ═══════════════════════════════════════════════════════════════

FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False
FLASK_ENV = 'production'

# ═══════════════════════════════════════════════════════════════
# DATABASE CONFIGURATION (Supabase / PostgreSQL with pgvector)
# ═══════════════════════════════════════════════════════════════

SUPABASE_URL = 'your_supabase_url'
SUPABASE_KEY = 'your_supabase_key'
SUPABASE_TIMEOUT = 30
VECTOR_DIMENSION = 768

# ═══════════════════════════════════════════════════════════════
# X.AI GROK CONFIGURATION (Vigil Knowledge Source)
# ═══════════════════════════════════════════════════════════════

XAI_API_KEY = 'your_xai_api_key'
XAI_API_URL = 'https://api.x.ai/v1/chat/completions'
XAI_MODEL = 'grok-2'
XAI_TIMEOUT = 30

# X.AI Request Parameters
XAI_TEMPERATURE_FORMAL = 0.3  # For structured analysis
XAI_TEMPERATURE_CREATIVE = 0.7  # For brainstorming
XAI_MAX_TOKENS_STANDARD = 1500
XAI_MAX_TOKENS_EXTENDED = 2500
XAI_MAX_TOKENS_SHORT = 500

# ═══════════════════════════════════════════════════════════════
# MODEL CONFIGURATION (DistilBERT for embeddings)
# ═══════════════════════════════════════════════════════════════

MODEL_NAME = 'dualpathtransformer'
EMBEDDING_MODEL = 'distilbert-base-uncased'
EMBEDDING_DIMENSION = 768
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══════════════════════════════════════════════════════════════
# SEARCH PARAMETERS (Supabase Vector Search)
# ═══════════════════════════════════════════════════════════════

VECTOR_MATCH_THRESHOLD = 0.7
VECTOR_MATCH_COUNT_DEFAULT = 10
VECTOR_MATCH_COUNT_QUICK = 5

# ═══════════════════════════════════════════════════════════════
# GOVERNANCE RULES (Private Source Policies)
# ═══════════════════════════════════════════════════════════════

POLICIES = {
    'supplier_concentration': {
        'green': 0.10,      # <10%: GREEN (good)
        'yellow': 0.15,     # 10-15%: YELLOW (acceptable)
        'orange': 0.25,     # 15-25%: ORANGE (concerning)
        'red': 1.0          # >25%: RED (critical)
    },
    'incident_frequency': {
        'baseline': 1.0,        # Expected: 1/year
        'yellow_threshold': 1.6,  # 1.6/year: YELLOW
        'red_threshold': 2.3      # 2.3+/year: RED (actual 2024)
    },
    'financial_impact': {
        'department_head': 1000000,      # <$1M: Department head approval
        'vp_approval': 5000000,          # $1-5M: VP approval
        'ceo_approval': 10000000,        # $5-10M: CEO approval
        'board_notification': 100000000  # >$10M: Board notification
    },
    'recovery_time': {
        'low': 7,      # <7 days: Low risk
        'medium': 14,  # 7-14 days: Medium risk
        'high': 21,    # 14-21 days: High risk
        'critical': 0  # >21 days: Critical
    }
}

# ═══════════════════════════════════════════════════════════════
# ALERT CONFIGURATION
# ═══════════════════════════════════════════════════════════════

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
    },
    'MEDIUM': {
        'channels': ['DASHBOARD', 'EMAIL_DIGEST'],
        'escalate_to': ['PROCUREMENT', 'OPERATIONS'],
        'escalate_within_hours': 72
    },
    'LOW': {
        'channels': ['DASHBOARD_WIDGET'],
        'escalate_to': ['MONITORING_TEAM'],
        'escalate_within_hours': 168  # 1 week
    }
}

# ═══════════════════════════════════════════════════════════════
# FORMAT DETECTION (Question Type → Response Format Mapping)
# ═══════════════════════════════════════════════════════════════

FORMAT_MAPPING = {
    'PATTERN_DETECTION': {
        'description': 'Is this a pattern? Are we seeing a trend? Is this accelerating?',
        'format': 'STRUCTURED_ANALYSIS',
        'structure': [
            'YES/NO answer (first line)',
            'Evidence (bulleted)',
            'Historical precedent (narrative)',
            'Confidence score (percentage)',
            'Implications (paragraphs)'
        ]
    },
    'ACTION_PLAN': {
        'description': 'What should we do? How do we fix this?',
        'format': 'PRIORITIZED_ACTIONS',
        'structure': [
            'Immediate (next 24-48 hours)',
            'Short-term (days 1-7)',
            'Medium-term (weeks 2-4)',
            'Long-term (months 2-6)',
            'Each with: [ACTION] | [OWNER] | [TIMELINE] | [IMPACT]'
        ]
    },
    'COMPARISON': {
        'description': 'Compare A to B? Which is worse?',
        'format': 'COMPARISON_MATRIX',
        'structure': [
            'Side-by-side table',
            'Scoring system (1-10)',
            'Key differentiators highlighted',
            'Winner/loser assessment',
            'Summary narrative'
        ]
    },
    'IMPACT_ASSESSMENT': {
        'description': 'How bad is this? What is the impact?',
        'format': 'IMPACT_PYRAMID',
        'structure': [
            'Top: Financial impact (clearest number)',
            'Middle: Operational impact (timeline, scope)',
            'Base: Strategic impact (long-term implications)',
            'Each level: detailed but progressive'
        ]
    },
    'ROOT_CAUSE_INVESTIGATION': {
        'description': 'What happened? Why did this occur?',
        'format': 'TIMELINE_AND_NARRATIVE',
        'structure': [
            'Timeline (chronological, clear dates)',
            'Root cause narrative (paragraphs)',
            'Contributing factors (bulleted)',
            'Preventability assessment'
        ]
    },
    'STRATEGIC_PLANNING': {
        'description': 'What is our risk profile? Where are we vulnerable?',
        'format': 'STRATEGIC_FRAMEWORK',
        'structure': [
            'Risk heat map (visual: high/medium/low)',
            'Vulnerability categories (by type, by geography)',
            'Trend indicators (improving/worsening)',
            'Strategic implications (narrative)'
        ]
    },
    'GENERAL_ANALYSIS': {
        'description': 'Default if none of above clearly fit',
        'format': 'STRUCTURED_INSTITUTIONAL',
        'structure': [
            'Classification',
            'Analysis',
            'Recommendations',
            'Governance'
        ]
    }
}

# ═══════════════════════════════════════════════════════════════
# SYNTHESIS CONFIGURATION
# ═══════════════════════════════════════════════════════════════

SYNTHESIS = {
    'show_source_attribution': True,    # Always show (Private Source) and (Vigil)
    'show_confidence_levels': True,     # Show confidence % for each source
    'show_consensus': True,             # Highlight when sources agree
    'default_confidence_private': 0.95, # Private Source baseline
    'default_confidence_vigil': 0.75,   # Vigil baseline
    'consensus_confidence_boost': 0.05  # Add 5% when both sources agree
}

# ═══════════════════════════════════════════════════════════════
# SCORING WEIGHTS (for solution/problem ranking)
# ═══════════════════════════════════════════════════════════════

SCORING_WEIGHTS = {
    'effectiveness': 0.35,       # How well it worked
    'applicability': 0.25,       # Does it apply here?
    'cost_efficiency': 0.15,     # Cost vs benefit
    'timeline': 0.15,            # Can we do it fast?
    'risk_level': 0.10           # How risky is it?
}

# ═══════════════════════════════════════════════════════════════
# CONVERSATION HISTORY
# ═══════════════════════════════════════════════════════════════

HISTORY_CONFIG = {
    'max_messages': 1000,
    'save_to_database': True,
    'include_metadata': True,
    'include_sources': True,
    'include_confidence': True
}

# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'vigil.log'
LOG_MAX_SIZE = 10485760  # 10MB
LOG_BACKUP_COUNT = 5

# ═══════════════════════════════════════════════════════════════
# PERFORMANCE TUNING
# ═══════════════════════════════════════════════════════════════

PERFORMANCE = {
    'batch_processing_enabled': True,
    'cache_embeddings': True,
    'async_alerts': True,
    'embedding_cache_ttl': 3600,  # 1 hour
    'search_timeout': 30,
    'grok_request_timeout': 30
}

# ═══════════════════════════════════════════════════════════════
# FEATURE FLAGS
# ═══════════════════════════════════════════════════════════════

FEATURES = {
    'dual_source_synthesis': True,
    'alert_generation': True,
    'format_detection': True,
    'solution_matching': True,
    'continuous_monitoring': True,
    'response_learning': True
}

# ═══════════════════════════════════════════════════════════════
# COMPANY DATA (for testing/demo)
# ═══════════════════════════════════════════════════════════════

COMPANY_DEMO_DATA = {
    'name': 'TechMfg Industries',
    'industry': 'Electronics Manufacturing',
    'headquarters': 'Austin, TX',
    'employees': 2400,
    'suppliers': 340,
    'regions': 18,
    
    'historical_incidents': {
        '2021': {'count': 1.0, 'avg_cost': 5800000, 'avg_recovery_days': 21},
        '2022': {'count': 1.5, 'avg_cost': 3100000, 'avg_recovery_days': 14},
        '2023': {'count': 1.6, 'avg_cost': 7200000, 'avg_recovery_days': 18},
        '2024': {'count': 2.3, 'avg_cost': 2100000, 'avg_recovery_days': 11}
    }
}
