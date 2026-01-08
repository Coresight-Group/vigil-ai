"""
VIGIL Flask Application - Risk Management API
Serves interconnected risk analysis with dual-source synthesis
"""

from flask import Flask, request, jsonify, g, send_from_directory
from flask_cors import CORS
import torch
import os
import json
import time
import logging
from datetime import datetime
import requests
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =========================================================================
# IMPORTS - Fixed ordering and error handling
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
    print(f"ERROR: Cannot import from main: {e}")
    print("Make sure main.py exists in the same directory")
    exit(1)

try:
    from utils import (
        create_synthesis_statement, determine_consensus_level,
        create_synthesis_alert, calculate_urgency_from_severity,
        get_escalation_path, get_notification_channels,
        calculate_combined_confidence, format_synthesis_section,
        validate_problem, validate_alert
    )
except ImportError:
    print("Warning: utils.py not found - some features unavailable")

try:
    import config
except ImportError:
    print("Warning: config.py not found - using environment defaults")
    config = None

# =========================================================================
# LOGGING SETUP
# =========================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================================
# FLASK APP INITIALIZATION
# =========================================================================

app = Flask(__name__)

# CORS configuration - Allow all origins for HF Spaces
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "max_age": 3600,
        "supports_credentials": False
    }
})

# =========================================================================
# ENVIRONMENT VARIABLES - Properly secured
# =========================================================================

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
XAI_API_KEY = os.getenv('XAI_API_KEY')
XAI_API_URL = 'https://api.x.ai/v1/chat/completions'
FLASK_ENV = os.getenv('FLASK_ENV', 'development')

# Validate required env vars
if not all([SUPABASE_URL, SUPABASE_KEY, XAI_API_KEY]):
    logger.warning("Missing required environment variables:")
    logger.warning(f"  SUPABASE_URL: {'✓' if SUPABASE_URL else '✗'}")
    logger.warning(f"  SUPABASE_KEY: {'✓' if SUPABASE_KEY else '✗'}")
    logger.warning(f"  XAI_API_KEY: {'✓' if XAI_API_KEY else '✗'}")

# =========================================================================
# DATABASE INITIALIZATION
# =========================================================================

SUPABASE_CONNECTED = False
supabase = None

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    # Test connection
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
    logger.info("  Components: GrokIntelligenceEngine, RiskCorrelationEngine,")
    logger.info("             SchemaValidator, DocumentStore")
except Exception as e:
    logger.error(f"✗ Error loading transformer: {e}")
    MODEL_LOADED = False

# =========================================================================
# REQUEST LOGGING & MONITORING
# =========================================================================

@app.before_request
def before_request():
    """Log incoming request"""
    g.start_time = time.time()
    logger.debug(f"{request.method} {request.path} from {request.remote_addr}")

@app.after_request
def after_request(response):
    """Log outgoing response"""
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
# SYSTEM PROMPTS
# =========================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are VIGIL, enterprise risk management analyst.

CRITICAL INSTRUCTION - DUAL-SOURCE SYNTHESIS WITH ATTRIBUTION:

You are receiving analysis from TWO SOURCES:
- (Private Source): Your company's actual incident history (Supabase)
- (Vigil): Industry knowledge and best practices (Grok research)

Your task: Synthesize both sources together, always citing where each came from.

FORMAT:

1. EXPLICIT PROBLEM ANSWERED
   - Direct answer to what user asked
   - Confidence level (high/medium/low)

2. EVIDENCE (FROM BOTH SOURCES)
   - Private Source evidence (our history)
   - Vigil evidence (industry knowledge)
   - Where they agree/disagree

3. ANALYSIS (WITH INTERCONNECTED RISK CONTEXT)
   - Self-conflicts identified
   - Historical precedents
   - Recurring patterns
   - Cascading effects on other risks
   - Timeline correlations

4. RECOMMENDATIONS (PRIORITIZED)
   - What to do (from proven solutions)
   - Timeline (immediate/short/long-term)
   - Success likelihood (based on history + industry)

5. GOVERNANCE ALERT
   - Policy violations if any
   - Escalation path
   - Decision authority needed
"""

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def format_interconnected_analysis(analysis):
    """Format interconnected risk analysis results for display"""
    
    formatted = "\n" + "="*60 + "\n"
    formatted += "INTERCONNECTED RISK ANALYSIS\n"
    formatted += "="*60 + "\n"
    
    # Self-conflicts
    if analysis.get('self_conflicts'):
        formatted += "\n⚠️ INTERNAL CONFLICTS DETECTED\n"
        for conflict in analysis['self_conflicts'][:2]:
            formatted += f"  • {conflict.get('description')}\n"
    
    # Historical matches
    if analysis.get('historical_matches'):
        formatted += f"\n📊 HISTORICAL PRECEDENTS ({len(analysis['historical_matches'])})\n"
        for match in analysis['historical_matches'][:2]:
            desc = match['original_description'][:80]
            sim = match.get('similarity_score', 0.0)
            date = match['original_date'][:10]
            formatted += f"  • {desc}...\n"
            formatted += f"    Similarity: {sim:.1%} | Date: {date}\n"
    
    # Recurring patterns
    if analysis.get('recurring_patterns'):
        pattern = analysis['recurring_patterns'][0]
        formatted += f"\n🔄 RECURRING PATTERN\n"
        formatted += f"  • {pattern['description']}\n"
        formatted += f"  • Severity: {pattern['severity'].upper()}\n"
    
    # Cascading effects
    if analysis.get('cascading_effects'):
        formatted += f"\n⚡ CASCADING EFFECTS ({len(analysis['cascading_effects'])})\n"
        for effect in analysis['cascading_effects'][:2]:
            affected = effect['affected_description'][:60]
            formatted += f"  • {effect['affected_risk_type']}: {affected}...\n"
    
    # Timeline correlation
    if analysis.get('timeline_correlations'):
        corr = analysis['timeline_correlations'][0]
        count = corr.get('related_events_count', 0)
        window = corr.get('window_days', 14)
        formatted += f"\n📅 TIMELINE CORRELATION\n"
        formatted += f"  • {count} related risks in {window}-day window\n"
    
    # Grok intelligence
    grok = analysis.get('grok_intelligence', {})
    if grok.get('context', {}).get('success'):
        findings = grok['context'].get('findings', '')[:200]
        formatted += f"\n🌐 INDUSTRY INTELLIGENCE (from Grok)\n"
        formatted += f"  {findings}...\n"
    
    formatted += "\n" + "="*60 + "\n"
    return formatted

# =========================================================================
# API ENDPOINTS
# =========================================================================

@app.route('/', methods=['GET'])
def root():
    """Serve index.html"""
    try:
        with open('index.html', 'r') as f:
            html_content = f.read()
            return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}
    except FileNotFoundError:
        return jsonify({
            'success': True,
            'message': 'VIGIL Risk Intelligence Platform',
            'version': '2.0',
            'status': 'running',
            'endpoints': {
                'health': '/api/health',
                'analyze': '/api/risks/analyze',
                'search': '/api/risks/search',
                'stats': '/api/stats',
                'chat': '/api/chat'
            }
        })

@app.route('/styles.css', methods=['GET'])
def serve_css():
    """Serve styles.css"""
    try:
        with open('styles.css', 'r') as f:
            css_content = f.read()
            return css_content, 200, {'Content-Type': 'text/css; charset=utf-8'}
    except FileNotFoundError:
        return 'Not found', 404

@app.route('/script.js', methods=['GET'])
def serve_js():
    """Serve script.js"""
    try:
        with open('script.js', 'r') as f:
            js_content = f.read()
            return js_content, 200, {'Content-Type': 'application/javascript; charset=utf-8'}
    except FileNotFoundError:
        return 'Not found', 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy' if (MODEL_LOADED and SUPABASE_CONNECTED) else 'degraded',
        'transformer_loaded': MODEL_LOADED,
        'supabase_connected': SUPABASE_CONNECTED,
        'message': 'VIGIL system operational',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/risks/analyze', methods=['POST'])
def analyze_risk():
    """
    Analyze risk with full interconnected analysis
    
    Request:
    {
        "description": "Risk description (20+ characters)"
    }
    
    Returns complete analysis including:
    - Classification (risk type, severity)
    - Historical context
    - Cascading effects
    - Industry intelligence
    """
    
    if not MODEL_LOADED:
        logger.error("Transformer not loaded")
        return jsonify({
            'success': False,
            'error': 'Transformer not loaded'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON body provided'
            }), 400
        
        # Accept both 'description' and 'message' fields
        description = data.get('description') or data.get('message', '').strip()
        
        if not description:
            return jsonify({
                'success': False,
                'error': 'Description or message required'
            }), 400
        
        if len(description) < 20:
            return jsonify({
                'success': False,
                'error': 'Description must be at least 20 characters'
            }), 400
        
        if len(description) > 5000:
            return jsonify({
                'success': False,
                'error': 'Description must be less than 5000 characters'
            }), 400
        
        # Forward pass with interconnected analysis
        logger.info(f"Analyzing risk: {description[:50]}...")
        
        with torch.no_grad():
            output = transformer.forward(
                data=description,
                analyze_interconnections=True
            )
        
        if not output.get('success'):
            logger.error(f"Analysis failed: {output.get('error')}")
            return jsonify({
                'success': False,
                'error': output.get('error', 'Analysis failed')
            }), 400
        
        # Extract key fields
        analysis = output.get('analysis', {})
        
        # Generate narrative
        output_with_description = output.copy()
        output_with_description['description'] = description
        narrative = transformer.generate_narrative(output_with_description)
        
        # Format interconnected analysis
        formatted_analysis = format_interconnected_analysis(analysis)
        
        logger.info(f"Analysis successful - Risk type: {output.get('risk_type')}, "
                   f"Severity: {output.get('severity')}")
        
        return jsonify({
            'success': True,
            'classification': {
                'risk_type': output.get('risk_type'),
                'severity': output.get('severity'),
                'confidence': float(output.get('confidence', 0))
            },
            'narrative': narrative,
            'formatted_analysis': formatted_analysis,
            'detailed_analysis': {
                'self_conflicts': analysis.get('self_conflicts'),
                'historical_matches': analysis.get('historical_matches'),
                'recurring_patterns': analysis.get('recurring_patterns'),
                'cascading_effects': analysis.get('cascading_effects'),
                'timeline_correlations': analysis.get('timeline_correlations'),
                'grok_intelligence': analysis.get('grok_intelligence')
            },
            'doc_id': output.get('doc_id')
        }), 201
        
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory")
        return jsonify({
            'success': False,
            'error': 'System memory exceeded'
        }), 503
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Analysis error: {str(e)}'
        }), 500

@app.route('/api/risks/search', methods=['GET'])
def search_risks():
    """
    Semantic search across all risks
    
    Query params:
    - q: search query (required)
    - top_k: number of results (default: 5, max: 20)
    """
    
    if not MODEL_LOADED:
        return jsonify({
            'success': False,
            'error': 'Transformer not loaded'
        }), 503
    
    try:
        query = request.args.get('q', '').strip()
        top_k = request.args.get('top_k', default=5, type=int)
        
        # Validate parameters
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query parameter required'
            }), 400
        
        if len(query) < 5:
            return jsonify({
                'success': False,
                'error': 'Query must be at least 5 characters'
            }), 400
        
        if top_k < 1 or top_k > 20:
            top_k = min(max(top_k, 1), 20)
        
        logger.info(f"Searching for: {query} (top_k={top_k})")
        
        results = transformer.semantic_search(query, top_k=top_k)
        
        logger.info(f"Found {len(results)} results")
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get vector store statistics"""
    
    if not MODEL_LOADED:
        return jsonify({
            'success': False,
            'error': 'Transformer not loaded'
        }), 503
    
    try:
        stats = transformer.get_vector_store_stats()
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint - routes to analyze_risk
    Accepts: {"message": "..."}
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Message required'
            }), 400
        
        # Route to analyze endpoint with 'message' field
        request.data = json.dumps({
            'description': data['message']
        }).encode()
        
        return analyze_risk()
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =========================================================================
# MAIN
# =========================================================================

if __name__ == '__main__':
    if FLASK_ENV == 'production':
        logger.warning("Running in PRODUCTION mode - use Gunicorn!")
        logger.warning("Recommended: gunicorn --workers=4 --bind=0.0.0.0:5000 app:app")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)