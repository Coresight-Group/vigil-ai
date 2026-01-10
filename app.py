"""
VIGIL Flask Application - Risk Management API
Handles Structured (Schema on Write), Semi-Structured (Hybrid), and 
Unstructured (Schema on Read) data with unified DistilBERT embeddings.

Integrates:
- Claude AI for deep reasoning and Vigil Summary
- Grok Intelligence as Agentic RAG processor
- pgvector for semantic search across all data types
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import torch
import os
import json
import time
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional
import requests
from supabase import create_client, Client
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# =========================================================================
# LOGGING SETUP
# =========================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================================
# SCHEMA VALIDATORS FOR ALL DATA TYPES
# =========================================================================

class ValidationError(Exception):
    """Custom validation error."""
    def __init__(self, field: str, reason: str, value: Any = None):
        self.field = field
        self.reason = reason
        self.value = value
        super().__init__(f"Validation Error in '{field}': {reason}")


class StructuredValidator:
    """Schema on Write - Strict validation for structured data."""
    
    MIN_LENGTH = 20
    MAX_LENGTH = 5000
    REQUIRED_FIELDS = ['description']
    VALID_RISK_TYPES = ['SUPPLY_CHAIN', 'QUALITY', 'DELIVERY', 'PRODUCTION', 'BRAND']
    VALID_SEVERITIES = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    def validate(self, data: Dict) -> Dict:
        """Validate structured data - REJECT if invalid."""
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in data or not data[field]:
                raise ValidationError(field, 'Required field missing', None)
        
        description = data.get('description', '').strip()
        
        # Validate length
        if len(description) < self.MIN_LENGTH:
            raise ValidationError('description', f'Too short (min {self.MIN_LENGTH})', len(description))
        
        if len(description) > self.MAX_LENGTH:
            raise ValidationError('description', f'Too long (max {self.MAX_LENGTH})', len(description))
        
        # Validate risk_type if provided
        if 'risk_type' in data and data['risk_type'] not in self.VALID_RISK_TYPES:
            raise ValidationError('risk_type', f'Invalid type: {data["risk_type"]}', data['risk_type'])
        
        # Validate severity if provided
        if 'severity' in data and data['severity'] not in self.VALID_SEVERITIES:
            raise ValidationError('severity', f'Invalid severity: {data["severity"]}', data['severity'])
        
        return {
            'description': description,
            'risk_type': data.get('risk_type', 'SUPPLY_CHAIN'),
            'severity': data.get('severity', 'MEDIUM'),
            'validated_at': datetime.now().isoformat(),
            'data_type': 'structured'
        }


class SemiStructuredValidator:
    """Hybrid Schema - Flexible validation with required core fields."""
    
    MIN_LENGTH = 20
    MAX_LENGTH = 5000
    REQUIRED_CORE = ['description']
    
    def validate(self, data: Dict) -> Dict:
        """Validate semi-structured data - Continue on partial errors."""
        description = data.get('description', '').strip()
        
        # Core validation (required)
        if not description:
            raise ValidationError('description', 'Core field missing', None)
        
        if len(description) < self.MIN_LENGTH or len(description) > self.MAX_LENGTH:
            raise ValidationError('description', 'Length out of range', len(description))
        
        # Optional fields with graceful handling
        validated = {
            'description': description,
            'validated_at': datetime.now().isoformat(),
            'data_type': 'semi-structured'
        }
        
        # Add optional structured fields if valid
        if 'risk_type' in data and data['risk_type']:
            validated['risk_type'] = data['risk_type']
        
        if 'severity' in data and data['severity']:
            validated['severity'] = data['severity']
        
        # Store flexible attributes in JSONB
        flexible_attrs = {}
        for key, value in data.items():
            if key not in ['description', 'risk_type', 'severity']:
                flexible_attrs[key] = value
        
        if flexible_attrs:
            validated['attributes'] = flexible_attrs
        
        return validated


class UnstructuredValidator:
    """Schema on Read - Graceful validation for unstructured data."""
    
    MIN_LENGTH = 20
    MAX_LENGTH = 5000
    
    def validate(self, text: str) -> Dict:
        """Validate unstructured data - Accept all, validate on read."""
        if isinstance(text, dict):
            text = text.get('description', '')
        
        text = str(text).strip()
        
        # Minimal validation - accept or continue with graceful degradation
        if not text:
            raise ValidationError('text', 'Empty input', None)
        
        if len(text) < self.MIN_LENGTH:
            logger.warning(f"Unstructured text below minimum length: {len(text)} chars")
        
        if len(text) > self.MAX_LENGTH:
            logger.warning(f"Unstructured text exceeds maximum: {len(text)} chars, truncating")
            text = text[:self.MAX_LENGTH]
        
        return {
            'description': text,
            'validated_at': datetime.now().isoformat(),
            'data_type': 'unstructured',
            'validation_strategy': 'schema_on_read',
            'partial_validation': len(text) < self.MIN_LENGTH or len(text) > self.MAX_LENGTH
        }


# =========================================================================
# IMPORTS
# =========================================================================

try:
    from main import (
        create_dual_path_transformer,
        DualPathRiskTransformer,
        GrokIntelligenceEngine,
        RiskCorrelationEngine,
        SchemaValidator,
        DocumentStore
    )
except ImportError as e:
    logger.error(f"Cannot import from main: {e}")
    exit(1)

# =========================================================================
# FLASK APP INITIALIZATION
# =========================================================================

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "max_age": 3600
    }
})

# =========================================================================
# ENVIRONMENT VARIABLES
# =========================================================================

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
XAI_API_KEY = os.getenv('XAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-opus-4-1')

# Initialize Claude client
try:
    claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    logger.info("✓ Claude AI client initialized")
except Exception as e:
    logger.warning(f"Claude AI initialization warning: {e}")
    claude_client = None

# =========================================================================
# DATABASE INITIALIZATION
# =========================================================================

SUPABASE_CONNECTED = False
supabase = None

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    health = supabase.table('risks').select('id').limit(1).execute()
    SUPABASE_CONNECTED = True
    logger.info("✓ Supabase connection successful")
except Exception as e:
    logger.error(f"✗ Supabase connection failed: {e}")
    SUPABASE_CONNECTED = False

# =========================================================================
# TRANSFORMER INITIALIZATION
# =========================================================================

MODEL_LOADED = False
transformer = None

try:
    transformer = create_dual_path_transformer(
        grok_api_key=XAI_API_KEY,
        supabase_client=supabase if SUPABASE_CONNECTED else None
    )
    MODEL_LOADED = True
    logger.info("✓ VIGIL Dual-Path Transformer Initialized")
except Exception as e:
    logger.error(f"✗ Error loading transformer: {e}")
    MODEL_LOADED = False

# =========================================================================
# REQUEST LOGGING
# =========================================================================

@app.before_request
def before_request():
    """Log incoming request."""
    g.start_time = time.time()

@app.after_request
def after_request(response):
    """Log outgoing response."""
    duration = time.time() - g.start_time
    logger.debug(f"Response: {response.status_code} ({duration:.3f}s)")
    return response

# =========================================================================
# ERROR HANDLERS
# =========================================================================

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'success': False, 'error': 'Bad request'}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# =========================================================================
# API ENDPOINTS
# =========================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'transformer_loaded': MODEL_LOADED,
        'supabase_connected': SUPABASE_CONNECTED,
        'message': 'VIGIL system operational',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/risks/analyze', methods=['POST'])
def analyze_risk():
    """
    Analyze risk with all data type support.
    
    Handles:
    - Structured (Schema on Write)
    - Semi-Structured (Hybrid)
    - Unstructured (Schema on Read)
    """
    
    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'Transformer not loaded'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body'}), 400
        
        description = data.get('description', '').strip()
        data_type = data.get('data_type', 'unstructured')  # Default to unstructured
        
        # ===== ROUTE TO APPROPRIATE VALIDATOR =====
        
        if data_type == 'structured':
            validator = StructuredValidator()
            schema_strategy = 'schema-on-write'
        elif data_type == 'semi-structured':
            validator = SemiStructuredValidator()
            schema_strategy = 'hybrid'
        else:  # unstructured
            validator = UnstructuredValidator()
            schema_strategy = 'schema-on-read'
        
        # Validate based on data type
        try:
            validated_input = validator.validate(data if data_type in ['structured', 'semi-structured'] else description)
            validation_status = 'validated'
        except ValidationError as e:
            if data_type == 'structured':
                # Schema on Write: REJECT
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'field': e.field,
                    'validation_info': {
                        'data_type': data_type,
                        'schema_strategy': schema_strategy,
                        'status': 'rejected'
                    }
                }), 400
            else:
                # Schema on Read / Hybrid: Continue with graceful degradation
                logger.warning(f"Validation warning ({data_type}): {e}")
                validated_input = {'description': description, 'data_type': data_type}
                validation_status = 'partial'
        
        logger.info(f"Analyzing {data_type} risk: {validated_input['description'][:50]}...")
        
        # ===== UNIFIED EMBEDDING & ANALYSIS =====
        
        with torch.no_grad():
            output = transformer.forward(
                data=validated_input['description'],
                analyze_interconnections=True
            )
        
        if not output.get('success'):
            return jsonify({
                'success': False,
                'error': output.get('error', 'Analysis failed'),
                'validation_info': {'data_type': data_type, 'schema_strategy': schema_strategy}
            }), 400
        
        # ===== CLAUDE AI VIGIL SUMMARY (all data types) =====
        
        vigil_summary = {}
        if claude_client:
            try:
                vigil_prompt = f"""Analyze this risk and provide a seamless Vigil Summary 
combining organizational insights with industry knowledge.

Risk: {validated_input['description']}
Type: {output.get('risk_type', 'SUPPLY_CHAIN')}
Data Source: {data_type}

Provide unified narrative:
{{ "situation": "...", "context": "...", "approach": "...", "timeline": "..." }}"""
                
                response = claude_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": vigil_prompt}]
                )
                
                try:
                    vigil_summary = json.loads(response.content[0].text)
                except json.JSONDecodeError:
                    match = re.search(r'\{.*\}', response.content[0].text, re.DOTALL)
                    vigil_summary = json.loads(match.group()) if match else {}
            except Exception as e:
                logger.warning(f"Claude processing warning: {e}")
        
        # ===== COMPLEXITY & ALERTS =====
        
        complexity_score = _calculate_complexity(validated_input['description'], output.get('severity'))
        alerts = _generate_alerts(complexity_score, output.get('risk_type'))
        solutions = _generate_solutions(complexity_score, output.get('risk_type'))
        
        # ===== GENERATE NARRATIVE =====
        
        output_with_description = output.copy()
        output_with_description['description'] = validated_input['description']
        narrative = transformer.generate_narrative(output_with_description)
        
        logger.info(f"Analysis successful - Type: {data_type}, Risk: {output.get('risk_type')}, "
                   f"Severity: {output.get('severity')}, Complexity: {complexity_score}/10")
        
        return jsonify({
            'success': True,
            'classification': {
                'risk_type': output.get('risk_type'),
                'severity': output.get('severity'),
                'confidence': float(output.get('confidence', 0)),
                'complexity_score': complexity_score
            },
            'vigil_summary': vigil_summary,
            'narrative': narrative,
            'detailed_analysis': {
                'self_conflicts': output.get('analysis', {}).get('self_conflicts'),
                'historical_matches': output.get('analysis', {}).get('historical_matches'),
                'cascading_effects': output.get('analysis', {}).get('cascading_effects'),
                'grok_intelligence': output.get('analysis', {}).get('grok_intelligence')
            },
            'alerts': alerts,
            'solutions': solutions,
            'doc_id': output.get('doc_id'),
            'validation_info': {
                'data_type_processed': data_type,
                'schema_strategy': schema_strategy,
                'validation_status': validation_status,
                'paths_used': ['structured', 'unstructured'],
                'distilbert_vectorized': True,
                'claude_integrated': claude_client is not None,
                'grok_processed': output.get('analysis', {}).get('grok_intelligence') is not None
            }
        }), 201
        
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory")
        return jsonify({'success': False, 'error': 'System memory exceeded'}), 503
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Analysis error: {str(e)}'}), 500


@app.route('/api/risks/search', methods=['GET'])
def search_risks():
    """Semantic search across all data types."""
    
    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'Transformer not loaded'}), 503
    
    try:
        query = request.args.get('q', '').strip()
        top_k = request.args.get('top_k', default=5, type=int)
        data_types = request.args.get('data_types', default='all')  # all, structured, unstructured, semi-structured
        
        if not query:
            return jsonify({'success': False, 'error': 'Query required'}), 400
        
        if len(query) < 5:
            return jsonify({'success': False, 'error': 'Query must be at least 5 chars'}), 400
        
        top_k = min(max(top_k, 1), 20)
        
        logger.info(f"Searching ({data_types}): {query}")
        
        results = transformer.semantic_search(query, top_k=top_k)
        
        return jsonify({
            'success': True,
            'query': query,
            'data_types_searched': data_types,
            'results': results,
            'count': len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    
    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'Transformer not loaded'}), 503
    
    try:
        stats = transformer.get_vector_store_stats()
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'data_types_supported': ['structured', 'semi-structured', 'unstructured'],
            'schema_strategies': {
                'structured': 'schema-on-write',
                'semi-structured': 'hybrid',
                'unstructured': 'schema-on-read'
            },
            'embedding_model': 'DistilBERT',
            'embedding_dimension': 768,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def _calculate_complexity(description: str, severity: str) -> int:
    """Calculate complexity (1-10)."""
    complexity = 1
    keywords = {
        3: ["crisis", "bankruptcy", "quit"],
        2: ["urgent", "multiple", "revenue"],
        1: ["emergency", "immediate"]
    }
    
    desc_lower = description.lower()
    for score, words in keywords.items():
        for word in words:
            if word in desc_lower:
                complexity += score
    
    severity_map = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
    complexity += severity_map.get(severity, 0)
    
    return max(1, min(10, complexity))


def _generate_alerts(complexity_score: int, risk_type: str) -> list:
    """Generate alerts (0-3)."""
    alerts = []
    
    if complexity_score <= 3:
        return alerts
    
    if 4 <= complexity_score <= 7:
        alerts.append({
            "alert_id": f"alert_{int(time.time())}",
            "alert_level": "HIGH" if complexity_score >= 6 else "MEDIUM",
            "title": f"{risk_type} Risk Alert",
            "recommendation": "Review Vigil Summary for action items"
        })
    
    elif complexity_score >= 8:
        alerts.append({
            "alert_id": f"alert_crisis_{int(time.time())}",
            "alert_level": "CRITICAL",
            "title": f"CRISIS: {risk_type}",
            "recommendation": "Establish crisis command immediately"
        })
        
        alerts.append({
            "alert_id": f"alert_coord_{int(time.time())}",
            "alert_level": "HIGH",
            "title": "Coordination Required",
            "recommendation": "Delegate authority and assign teams"
        })
    
    return alerts


def _generate_solutions(complexity_score: int, risk_type: str) -> list:
    """Generate solutions (1-7)."""
    solutions = []
    
    if complexity_score <= 3:
        solutions.append({
            "tier": 1,
            "title": "Primary Action",
            "success_probability": 0.85
        })
    
    elif 4 <= complexity_score <= 7:
        for i, (title, prob) in enumerate([
            ("Primary Strategy", 0.75),
            ("Backup Approach", 0.80),
            ("VP Coordination", 0.70)
        ]):
            solutions.append({
                "tier": 1 if i == 0 else 2,
                "title": title,
                "success_probability": prob
            })
    
    elif complexity_score >= 8:
        for tier, (title, prob) in enumerate([
            ("Crisis Command", 0.99),
            ("Emergency Sourcing", 0.85),
            ("Contract Manufacturing", 0.92),
            ("Customer Communication", 0.80),
            ("Production Scheduling", 0.88),
            ("Dual Sourcing Strategy", 0.95)
        ], 1):
            solutions.append({
                "tier": min((tier - 1) // 2 + 1, 4),
                "title": f"TIER {min((tier - 1) // 2 + 1, 4)}: {title}",
                "success_probability": prob
            })
    
    return solutions


# =========================================================================
# MAIN
# =========================================================================

if __name__ == '__main__':
    if FLASK_ENV == 'production':
        logger.warning("Use Gunicorn in production!")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)