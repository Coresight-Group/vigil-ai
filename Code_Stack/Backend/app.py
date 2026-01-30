"""
VIGIL Flask Application - Risk Management API
Handles Structured (Schema on Write), Semi-Structured (Hybrid), and
Unstructured (Schema on Read) data with unified DistilBERT embeddings.

Integrates:
- Grok Intelligence for deep reasoning, Vigil Summary, and Agentic RAG
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
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
GROK_MODEL = os.getenv('GROK_MODEL', 'grok-3')
GROK_API_URL = 'https://api.x.ai/v1/chat/completions'

# Initialize Grok client configuration
grok_configured = False
if XAI_API_KEY:
    grok_configured = True
    logger.info("✓ Grok AI client configured")
else:
    logger.warning("Grok AI not configured - XAI_API_KEY not set")


def _call_grok_api(
    prompt: str,
    system_message: str = "You are Vigil, an intelligent risk management assistant powered by Grok.",
    max_tokens: int = 1500,
    temperature: float = 0.3,
    timeout: int = 30
) -> Optional[str]:
    """
    Call Grok API for AI-powered analysis.

    Args:
        prompt: The user prompt to send
        system_message: System context for Grok
        max_tokens: Maximum response tokens
        temperature: Creativity level (0.0-1.0)
        timeout: Request timeout in seconds

    Returns:
        Grok's response text or None if failed
    """
    if not XAI_API_KEY:
        logger.warning("Grok API key not configured")
        return None

    try:
        response = requests.post(
            GROK_API_URL,
            headers={
                'Authorization': f'Bearer {XAI_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': GROK_MODEL,
                'messages': [
                    {'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': max_tokens,
                'temperature': temperature
            },
            timeout=timeout
        )

        if response.status_code == 200:
            data = response.json()
            return data.get('choices', [{}])[0].get('message', {}).get('content', '')
        else:
            logger.warning(f"Grok API error: {response.status_code} - {response.text[:200]}")
            return None

    except requests.exceptions.Timeout:
        logger.warning("Grok API timeout")
        return None
    except Exception as e:
        logger.warning(f"Grok API error: {e}")
        return None


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
        # Now supports dict input, structured_data, and handles long text via chunking

        with torch.no_grad():
            output = transformer.forward(
                data=validated_input,  # Pass full validated input (dict with structured fields)
                analyze_interconnections=True,
                structured_data=validated_input.get('attributes')  # Pass JSONB attributes if any
            )
        
        if not output.get('success'):
            return jsonify({
                'success': False,
                'error': output.get('error', 'Analysis failed'),
                'validation_info': {'data_type': data_type, 'schema_strategy': schema_strategy}
            }), 400
        
        # ===== GROK AI VIGIL SUMMARY (all data types) =====

        vigil_summary = {}
        if grok_configured:
            try:
                vigil_prompt = f"""Analyze this risk and provide a seamless Vigil Summary
combining organizational insights with industry knowledge.

Risk: {validated_input['description']}
Type: {output.get('risk_type', 'SUPPLY_CHAIN')}
Data Source: {data_type}

Provide a unified narrative as JSON with these fields:
- situation: Current state and immediate concerns
- context: Industry context and relevant trends (include industry-specific insights)
- approach: Recommended mitigation approach with actionable steps
- timeline: Suggested timeline for action

Return ONLY valid JSON:
{{ "situation": "...", "context": "...", "approach": "...", "timeline": "..." }}"""

                grok_response = _call_grok_api(
                    prompt=vigil_prompt,
                    system_message="You are Vigil's core intelligence engine powered by Grok. Provide expert risk analysis combining industry knowledge with organizational insights.",
                    max_tokens=1500,
                    temperature=0.3
                )

                if grok_response:
                    try:
                        vigil_summary = json.loads(grok_response)
                    except json.JSONDecodeError:
                        match = re.search(r'\{.*\}', grok_response, re.DOTALL)
                        vigil_summary = json.loads(match.group()) if match else {}
            except Exception as e:
                logger.warning(f"Grok processing warning: {e}")
        
        # ===== COMPLEXITY & ALERTS =====

        complexity_score = _calculate_complexity(validated_input['description'], output.get('severity'))

        # Get Grok intelligence from output for alert synthesis
        grok_intelligence = output.get('analysis', {}).get('grok_intelligence')

        # Generate alerts using Grok/Supabase synthesis (same as output prompts, condensed)
        # Alerts are generated for EVERY risk, regardless of complexity
        alerts = _generate_alerts(
            description=validated_input['description'],
            risk_type=output.get('risk_type', 'SUPPLY_CHAIN'),
            severity=output.get('severity', 'MEDIUM'),
            vigil_summary=vigil_summary,
            grok_intelligence=grok_intelligence,
            supabase_client=supabase if SUPABASE_CONNECTED else None
        )

        # ===== DUAL-SOURCE SOLUTIONS (Supabase + Grok) =====
        # Generate solutions from both private data and external intelligence simultaneously
        dual_source_solutions = _generate_dual_source_solutions(
            transformer=transformer,
            risk_type=output.get('risk_type'),
            severity=output.get('severity'),
            description=validated_input['description'],
            complexity_score=complexity_score,
            structured_data=validated_input.get('attributes')
        )

        # Format solutions for API response with (Recommended) tags
        formatted_solutions = _format_solutions_for_response(dual_source_solutions)

        # Keep legacy solutions for backward compatibility
        legacy_solutions = _generate_solutions(complexity_score, output.get('risk_type'))
        
        # ===== GENERATE NARRATIVE =====
        
        output_with_description = output.copy()
        output_with_description['description'] = validated_input['description']
        narrative = transformer.generate_narrative(output_with_description)
        
        logger.info(f"Analysis successful - Type: {data_type}, Risk: {output.get('risk_type')}, "
                   f"Severity: {output.get('severity')}, Complexity: {complexity_score}/10")
        
        # ===== GENERATE FOLLOW-UP QUESTIONS =====
        # Vigil proactively asks follow-up questions when analyzing alerts
        follow_up_questions = _generate_follow_up_questions(
            risk_type=output.get('risk_type'),
            severity=output.get('severity'),
            vigil_summary=vigil_summary,
            grok_intelligence=grok_intelligence,
            alerts=alerts
        )

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
            # NEW: Dual-source solutions with tiering and source attribution
            'solutions': formatted_solutions,
            # Legacy solutions preserved for backward compatibility
            'legacy_solutions': legacy_solutions,
            # NEW: Follow-up questions for deeper analysis
            'follow_up_questions': follow_up_questions,
            'doc_id': output.get('doc_id'),
            'validation_info': {
                'data_type_processed': data_type,
                'schema_strategy': schema_strategy,
                'validation_status': validation_status,
                'paths_used': ['structured', 'unstructured'],
                'distilbert_vectorized': True,
                'grok_integrated': grok_configured,
                'grok_processed': output.get('analysis', {}).get('grok_intelligence') is not None,
                'dual_source_solutions': formatted_solutions.get('has_solutions', False),
                'solution_sources': {
                    'private_data': formatted_solutions.get('summary', {}).get('from_private_data', 0),
                    'external_intelligence': formatted_solutions.get('summary', {}).get('from_external_intelligence', 0)
                },
                'chunking_info': output.get('chunking_info', {}),
                'structured_data_used': output.get('structured_data', {})
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
    """
    Semantic search across all data types using pgvector.

    Query params:
        q: Search query (required)
        top_k: Number of results (default 5, max 20)
        risk_type: Filter by risk type (optional)
        severity: Filter by severity (optional)
    """

    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'Transformer not loaded'}), 503

    try:
        query = request.args.get('q', '').strip()
        top_k = request.args.get('top_k', default=5, type=int)
        filter_risk_type = request.args.get('risk_type')
        filter_severity = request.args.get('severity')

        if not query:
            return jsonify({'success': False, 'error': 'Query required'}), 400

        if len(query) < 5:
            return jsonify({'success': False, 'error': 'Query must be at least 5 chars'}), 400

        top_k = min(max(top_k, 1), 20)

        logger.info(f"Searching: {query} (top_k={top_k}, risk_type={filter_risk_type}, severity={filter_severity})")

        results = transformer.semantic_search(
            query=query,
            top_k=top_k,
            filter_risk_type=filter_risk_type,
            filter_severity=filter_severity
        )

        return jsonify({
            'success': True,
            'query': query,
            'filters': {
                'risk_type': filter_risk_type,
                'severity': filter_severity
            },
            'results': results,
            'count': len(results)
        }), 200

    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def conversational_chat():
    """
    Conversational chat endpoint powered by Grok for industry context.

    Handles follow-up questions with:
    - Industry-specific responses from Grok
    - References to user's own data in parentheses (from Supabase)
    - Contextual awareness of previous analysis

    Request body:
    {
        "message": "User's follow-up question",
        "context": {
            "risk_type": "SUPPLY_CHAIN",
            "severity": "HIGH",
            "previous_analysis": {...}  // Optional: previous vigil_summary
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body'}), 400

        message = data.get('message', '').strip()
        context = data.get('context', {})

        if not message:
            return jsonify({'success': False, 'error': 'Message required'}), 400

        if len(message) < 3:
            return jsonify({'success': False, 'error': 'Message too short'}), 400

        risk_type = context.get('risk_type', 'SUPPLY_CHAIN')
        severity = context.get('severity', 'MEDIUM')
        previous_analysis = context.get('previous_analysis', {})

        # ===== QUERY SUPABASE FOR RELEVANT DATA REFERENCES =====
        supabase_references = []
        if SUPABASE_CONNECTED and supabase:
            try:
                # Find related risks for context
                related = supabase.table('risks').select(
                    'id, description, risk_type, severity'
                ).eq('risk_type', risk_type).limit(5).execute()

                if related.data:
                    for r in related.data:
                        supabase_references.append({
                            'id': r['id'],
                            'description': r.get('description', '')[:100],
                            'severity': r.get('severity')
                        })
            except Exception as e:
                logger.warning(f"Supabase reference lookup: {e}")

        # ===== BUILD GROK CONVERSATIONAL PROMPT =====
        # Format Supabase references for the prompt
        data_refs = ""
        data_ref_count = 0
        if supabase_references:
            data_ref_count = len(supabase_references)
            data_refs = "\n\nRelevant data from user's database:\n"
            for ref in supabase_references[:3]:
                data_refs += f"- {ref['description']} [{ref['severity']}]\n"

        grok_prompt = f"""You are Vigil's conversational intelligence assistant powered by Grok. Have a natural, helpful conversation with the user about their risk management questions.

User's Question: {message}

Context:
- Risk Type: {risk_type}
- Severity: {severity}
- Previous Analysis: {json.dumps(previous_analysis)[:500] if previous_analysis else 'None'}
{data_refs}

Instructions for your response:
1. Be conversational and natural - respond like a knowledgeable colleague, not a formal report
2. Provide industry-specific insights with current trends and best practices
3. When referencing the user's own data, put it in parentheses like (based on your {data_ref_count} similar historical incidents) or (according to your risk history)
4. If the question is general industry knowledge (not specifically about their data), lead with industry insights and add relevant user data references in parentheses where applicable
5. Be helpful, specific, and actionable
6. Feel free to share relevant industry statistics, benchmarks, or common approaches
7. If you notice patterns or concerns, mention them conversationally

Respond naturally and conversationally - this is a dialogue, not a report."""

        # ===== CALL GROK API =====
        grok_response = _call_grok_api(
            prompt=grok_prompt,
            system_message="You are Vigil, an intelligent and conversational risk management assistant powered by Grok. You have deep industry knowledge and help users understand and manage their risks through natural dialogue. Be helpful, insightful, and conversational.",
            max_tokens=1000,
            temperature=0.7
        )

        if not grok_response:
            grok_response = f"I understand you're asking about {risk_type} risks. Based on industry standards, I'd recommend reviewing your historical data and current mitigation strategies. Could you provide more specific details about what aspect you'd like to explore?"

        # ===== GENERATE FOLLOW-UP SUGGESTIONS =====
        follow_ups = []
        if grok_configured:
            try:
                follow_up_prompt = f"""Based on this conversation, suggest 2-3 natural follow-up questions the user might want to ask next.

User asked: {message}
Response given: {grok_response[:400]}
Risk context: {risk_type} at {severity} severity

Generate conversational follow-up questions that would help the user:
- Dig deeper into the topic
- Explore related areas
- Take action on the advice given

Return ONLY a JSON array of strings (the questions):
["question 1", "question 2", "question 3"]"""

                follow_response = _call_grok_api(
                    prompt=follow_up_prompt,
                    system_message="Generate helpful follow-up questions for a risk management conversation.",
                    max_tokens=200,
                    temperature=0.6
                )

                if follow_response:
                    try:
                        follow_ups = json.loads(follow_response)
                    except:
                        match = re.search(r'\[.*\]', follow_response, re.DOTALL)
                        follow_ups = json.loads(match.group()) if match else []
            except Exception as e:
                logger.warning(f"Follow-up generation: {e}")

        return jsonify({
            'success': True,
            'response': grok_response,
            'source': 'grok',
            'data_references': supabase_references,
            'data_references_count': len(supabase_references),
            'follow_up_suggestions': follow_ups[:3],
            'context': {
                'risk_type': risk_type,
                'severity': severity
            }
        }), 200

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/risks/<risk_id>', methods=['GET'])
def get_risk(risk_id):
    """Get a specific risk by ID."""

    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'Transformer not loaded'}), 503

    try:
        result = transformer.get_risk(risk_id)

        if not result:
            return jsonify({'success': False, 'error': 'Risk not found'}), 404

        return jsonify({
            'success': True,
            'risk': result
        }), 200

    except Exception as e:
        logger.error(f"Get risk error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/risks/<risk_id>', methods=['DELETE'])
def delete_risk(risk_id):
    """Delete a risk by ID."""

    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'Transformer not loaded'}), 503

    try:
        deleted = transformer.delete_risk(risk_id)

        if deleted:
            return jsonify({
                'success': True,
                'message': f'Risk {risk_id} deleted'
            }), 200
        else:
            return jsonify({'success': False, 'error': 'Risk not found or delete failed'}), 404

    except Exception as e:
        logger.error(f"Delete risk error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/risks/recent', methods=['GET'])
def get_recent_risks():
    """Get most recent risks."""

    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'Transformer not loaded'}), 503

    try:
        limit = request.args.get('limit', default=10, type=int)
        limit = min(max(limit, 1), 50)

        results = transformer.get_recent_risks(limit)

        return jsonify({
            'success': True,
            'risks': results,
            'count': len(results)
        }), 200

    except Exception as e:
        logger.error(f"Get recent risks error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics including storage info."""

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
            'storage': {
                'persistent_enabled': stats.get('persistent_storage_enabled', False),
                'type': 'supabase_pgvector' if stats.get('persistent_storage_enabled') else 'in_memory',
                'supabase_connected': SUPABASE_CONNECTED
            },
            'embedding_model': 'DistilBERT',
            'embedding_dimension': 768,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================================================================
# PROCUREMENT ENDPOINTS
# =========================================================================

@app.route('/api/procurement/suggest', methods=['POST'])
def get_procurement_suggestions():
    """
    Get procurement suggestions for a specific risk scenario.

    Request body:
    {
        "risk_type": "PRODUCTION",
        "severity": "HIGH",
        "description": "Main assembly line motor showing signs of failure"
    }
    """
    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'Transformer not loaded'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body'}), 400

        risk_type = data.get('risk_type', 'SUPPLY_CHAIN')
        severity = data.get('severity', 'MEDIUM')
        description = data.get('description', '')

        if not description or len(description) < 10:
            return jsonify({'success': False, 'error': 'Description required (min 10 chars)'}), 400

        suggestions = transformer.procurement_engine.generate_suggestions(
            risk_type=risk_type,
            severity=severity,
            description=description,
            structured_data=data.get('structured_data')
        )

        return jsonify({
            'success': True,
            'risk_type': risk_type,
            'severity': severity,
            'suggestions': suggestions
        }), 200

    except Exception as e:
        logger.error(f"Procurement suggestion error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/procurement/history', methods=['GET'])
def get_procurement_history():
    """Get procurement suggestion history."""
    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'Transformer not loaded'}), 503

    if not SUPABASE_CONNECTED:
        return jsonify({'success': False, 'error': 'Database not connected'}), 503

    try:
        status_filter = request.args.get('status')  # suggested, approved, ordered, delivered
        limit = request.args.get('limit', default=20, type=int)
        limit = min(max(limit, 1), 100)

        query = supabase.table('procurement_history').select(
            'id, risk_id, item_id, supplier_id, quantity, status, '
            'suggested_at, approved_at, ordered_at, delivered_at'
        )

        if status_filter:
            query = query.eq('status', status_filter)

        result = query.order('suggested_at', desc=True).limit(limit).execute()

        return jsonify({
            'success': True,
            'history': result.data if result.data else [],
            'count': len(result.data) if result.data else 0
        }), 200

    except Exception as e:
        logger.error(f"Procurement history error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/procurement/<suggestion_id>/status', methods=['PUT'])
def update_procurement_status(suggestion_id):
    """
    Update procurement suggestion status.

    Request body:
    {
        "status": "approved"  // suggested, approved, ordered, delivered
    }
    """
    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'Transformer not loaded'}), 503

    try:
        data = request.get_json()
        if not data or 'status' not in data:
            return jsonify({'success': False, 'error': 'Status required'}), 400

        status = data['status']
        valid_statuses = ['suggested', 'approved', 'ordered', 'delivered', 'cancelled']

        if status not in valid_statuses:
            return jsonify({
                'success': False,
                'error': f'Invalid status. Must be one of: {valid_statuses}'
            }), 400

        updated = transformer.procurement_engine.update_suggestion_status(suggestion_id, status)

        if updated:
            return jsonify({
                'success': True,
                'message': f'Status updated to {status}'
            }), 200
        else:
            return jsonify({'success': False, 'error': 'Update failed'}), 500

    except Exception as e:
        logger.error(f"Update procurement status error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/suppliers', methods=['GET'])
def get_suppliers():
    """Get list of suppliers with optional filtering."""
    if not SUPABASE_CONNECTED:
        return jsonify({'success': False, 'error': 'Database not connected'}), 503

    try:
        category = request.args.get('category')
        limit = request.args.get('limit', default=20, type=int)

        query = supabase.table('suppliers').select(
            'id, name, category, lead_time_days, reliability_score, '
            'is_backup, pricing_tier, certifications'
        )

        if category:
            query = query.eq('category', category)

        result = query.order('reliability_score', desc=True).limit(limit).execute()

        return jsonify({
            'success': True,
            'suppliers': result.data if result.data else [],
            'count': len(result.data) if result.data else 0
        }), 200

    except Exception as e:
        logger.error(f"Get suppliers error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/documents/search', methods=['GET'])
def search_documents():
    """
    Search raw data documents in Supabase.

    Query params:
        q: Search query (required)
        limit: Number of results (default 10, max 50)
        doc_type: Filter by document type (optional)
    """
    if not SUPABASE_CONNECTED:
        return jsonify({'success': False, 'error': 'Database not connected'}), 503

    try:
        query = request.args.get('q', '').strip()
        limit = request.args.get('limit', default=10, type=int)
        doc_type = request.args.get('doc_type')

        if not query:
            return jsonify({'success': False, 'error': 'Query required'}), 400

        if len(query) < 2:
            return jsonify({'success': False, 'error': 'Query must be at least 2 chars'}), 400

        limit = min(max(limit, 1), 50)

        logger.info(f"Document search: {query} (limit={limit}, doc_type={doc_type})")

        # Search across multiple document tables
        results = []

        # Search in risks table
        try:
            risks_query = supabase.table('risks').select(
                'id, description, risk_type, severity, created_at'
            ).ilike('description', f'%{query}%').limit(limit)

            risks_result = risks_query.execute()
            if risks_result.data:
                for r in risks_result.data:
                    results.append({
                        'id': r['id'],
                        'title': r['description'][:100] + '...' if len(r.get('description', '')) > 100 else r.get('description', 'Untitled'),
                        'type': 'risk',
                        'icon': 'alert',
                        'metadata': {
                            'risk_type': r.get('risk_type'),
                            'severity': r.get('severity')
                        },
                        'created_at': r.get('created_at')
                    })
        except Exception as e:
            logger.warning(f"Risks search warning: {e}")

        # Search in solutions table if it exists
        try:
            solutions_query = supabase.table('solutions').select(
                'id, title, description, solution_category, created_at'
            ).or_(f'title.ilike.%{query}%,description.ilike.%{query}%').limit(limit)

            solutions_result = solutions_query.execute()
            if solutions_result.data:
                for s in solutions_result.data:
                    results.append({
                        'id': s['id'],
                        'title': s.get('title', 'Untitled Solution'),
                        'type': 'solution',
                        'icon': 'lightbulb',
                        'metadata': {
                            'category': s.get('solution_category')
                        },
                        'created_at': s.get('created_at')
                    })
        except Exception as e:
            logger.debug(f"Solutions table search: {e}")

        # Search in suppliers table
        try:
            suppliers_query = supabase.table('suppliers').select(
                'id, name, category, certifications, reliability_score'
            ).ilike('name', f'%{query}%').limit(limit)

            suppliers_result = suppliers_query.execute()
            if suppliers_result.data:
                for sup in suppliers_result.data:
                    results.append({
                        'id': sup['id'],
                        'title': sup.get('name', 'Unknown Supplier'),
                        'type': 'supplier',
                        'icon': 'building',
                        'metadata': {
                            'category': sup.get('category'),
                            'reliability': sup.get('reliability_score')
                        },
                        'created_at': None
                    })
        except Exception as e:
            logger.debug(f"Suppliers table search: {e}")

        # Search in inventory_items table
        try:
            inventory_query = supabase.table('inventory_items').select(
                'id, name, sku, category, current_stock, is_critical'
            ).or_(f'name.ilike.%{query}%,sku.ilike.%{query}%').limit(limit)

            inventory_result = inventory_query.execute()
            if inventory_result.data:
                for inv in inventory_result.data:
                    results.append({
                        'id': inv['id'],
                        'title': inv.get('name', 'Unknown Item'),
                        'type': 'inventory',
                        'icon': 'package',
                        'metadata': {
                            'sku': inv.get('sku'),
                            'category': inv.get('category'),
                            'stock': inv.get('current_stock'),
                            'critical': inv.get('is_critical')
                        },
                        'created_at': None
                    })
        except Exception as e:
            logger.debug(f"Inventory table search: {e}")

        # Filter by doc_type if specified
        if doc_type:
            results = [r for r in results if r['type'] == doc_type]

        # Sort by relevance (exact matches first) and limit
        results = sorted(results, key=lambda x: (
            0 if query.lower() in (x.get('title') or '').lower()[:50] else 1,
            x.get('created_at') or ''
        ), reverse=False)[:limit]

        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results),
            'doc_types': ['risk', 'solution', 'supplier', 'inventory']
        }), 200

    except Exception as e:
        logger.error(f"Document search error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/inventory', methods=['GET'])
def get_inventory():
    """Get inventory items with optional filtering."""
    if not SUPABASE_CONNECTED:
        return jsonify({'success': False, 'error': 'Database not connected'}), 503

    try:
        category = request.args.get('category')
        low_stock = request.args.get('low_stock', default='false').lower() == 'true'
        critical_only = request.args.get('critical_only', default='false').lower() == 'true'
        limit = request.args.get('limit', default=50, type=int)

        query = supabase.table('inventory_items').select(
            'id, name, sku, category, current_stock, reorder_point, '
            'reorder_quantity, unit_cost, lead_time_days, is_critical'
        )

        if category:
            query = query.eq('category', category)

        if critical_only:
            query = query.eq('is_critical', True)

        result = query.order('name').limit(limit).execute()

        items = result.data if result.data else []

        # Filter for low stock if requested
        if low_stock:
            items = [i for i in items if i.get('current_stock', 0) <= i.get('reorder_point', 0)]

        return jsonify({
            'success': True,
            'items': items,
            'count': len(items)
        }), 200

    except Exception as e:
        logger.error(f"Get inventory error: {str(e)}")
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


def _generate_alerts(
    description: str,
    risk_type: str,
    severity: str,
    vigil_summary: Dict = None,
    grok_intelligence: Dict = None,
    supabase_client: Client = None
) -> list:
    """
    Generate alerts by having Grok analyze Supabase data to find potential risks.

    Alerts work exactly like output prompts (Vigil Summary) but are condensed versions:
    1. Query Supabase for related historical risks and patterns
    2. Compare current risk with up-to-date industry issues (via Grok)
    3. Identify risks from Supabase data itself
    4. Use Grok to synthesize into condensed alerts

    This runs for EVERY risk analysis, regardless of complexity.
    """
    alerts = []
    alert_timestamp = int(time.time())

    # ===== STEP 1: Query Supabase for related risks and patterns =====
    supabase_risks = []
    supabase_patterns = []

    if supabase_client:
        try:
            # Find similar historical risks
            risks_query = supabase_client.table('risks').select(
                'id, description, risk_type, severity, created_at'
            ).eq('risk_type', risk_type).limit(10).execute()

            if risks_query.data:
                supabase_risks = risks_query.data

            # Find any patterns or recurring issues
            pattern_query = supabase_client.table('risks').select(
                'risk_type, severity, count'
            ).eq('severity', severity).limit(5).execute()

            if pattern_query.data:
                supabase_patterns = pattern_query.data

        except Exception as e:
            logger.warning(f"Supabase query for alerts: {e}")

    # ===== STEP 2: Build context from all sources =====
    # Extract Vigil Summary insights (from Grok's main analysis)
    vigil_situation = ""
    vigil_approach = ""
    if vigil_summary:
        vigil_situation = vigil_summary.get('situation') or vigil_summary.get('current_situation', '')
        vigil_approach = vigil_summary.get('approach') or vigil_summary.get('proven_approaches', '')

    # Extract Grok Intelligence insights (industry context)
    grok_context = ""
    grok_practices = ""
    if grok_intelligence:
        grok_context = grok_intelligence.get('industry_context') or grok_intelligence.get('context', '')
        practices = grok_intelligence.get('best_practices') or grok_intelligence.get('recommendations', [])
        if isinstance(practices, list) and practices:
            grok_practices = practices[0] if isinstance(practices[0], str) else str(practices[0])
        elif isinstance(practices, str):
            grok_practices = practices

    # ===== STEP 3: Use Grok to generate condensed alerts =====
    if grok_configured:
        try:
            # Build Supabase context string
            supabase_context = ""
            if supabase_risks:
                supabase_context = f"Historical risks from database ({len(supabase_risks)} found):\n"
                for r in supabase_risks[:5]:
                    supabase_context += f"- [{r.get('severity')}] {r.get('description', '')[:100]}\n"

            alert_prompt = f"""Analyze this risk and generate 1-3 concise alerts based on:
1. The current risk being analyzed
2. Historical patterns from our database (Supabase)
3. Current industry issues and trends

Current Risk:
- Description: {description[:500]}
- Type: {risk_type}
- Severity: {severity}

Vigil Analysis Summary:
- Situation: {vigil_situation[:300] if vigil_situation else 'Not available'}
- Recommended Approach: {vigil_approach[:300] if vigil_approach else 'Not available'}

Industry Intelligence:
- Context: {grok_context[:300] if grok_context else 'Not available'}
- Best Practices: {grok_practices[:300] if grok_practices else 'Not available'}

{supabase_context}

Generate 1-3 alerts as JSON array. Each alert should be a CONDENSED insight (max 100 chars for title, max 200 chars for summary).
Focus on:
- Risks identified by comparing with Supabase historical data
- Risks identified from current industry issues
- Actionable recommendations

Return ONLY valid JSON array:
[{{"alert_type": "supabase_pattern|industry_trend|combined", "title": "...", "summary": "...", "action": "...", "priority": "HIGH|MEDIUM|LOW"}}]"""

            response_text = _call_grok_api(
                prompt=alert_prompt,
                system_message="You are Vigil's alert generation engine powered by Grok. Generate concise, actionable alerts based on risk analysis.",
                max_tokens=800,
                temperature=0.3
            )

            if response_text:
                # Parse Grok's response
                try:
                    parsed_alerts = json.loads(response_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    parsed_alerts = json.loads(match.group()) if match else []

                # Format alerts with proper structure
                for i, alert_data in enumerate(parsed_alerts[:3]):
                    alert_type = alert_data.get('alert_type', 'combined')

                    # Determine source based on alert type
                    source = {
                        "supabase": alert_type in ['supabase_pattern', 'combined'],
                        "grok_intelligence": alert_type in ['industry_trend', 'combined'],
                        "vigil_summary": bool(vigil_summary)
                    }

                    # Map priority to alert level
                    priority = alert_data.get('priority', 'MEDIUM')
                    alert_level = priority if priority in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] else 'MEDIUM'

                    alerts.append({
                        "alert_id": f"alert_{alert_timestamp}_{i}",
                        "alert_level": alert_level,
                        "alert_type": alert_type,
                        "title": alert_data.get('title', f'{risk_type} Alert')[:100],
                        "summary": alert_data.get('summary', '')[:200],
                        "action": alert_data.get('action', 'Review full Vigil Summary for details.')[:200],
                        "source": source,
                        "data_sources": {
                            "supabase_risks_analyzed": len(supabase_risks),
                            "vigil_summary_used": bool(vigil_summary),
                            "grok_intelligence_used": bool(grok_intelligence)
                        }
                    })

        except Exception as e:
            logger.warning(f"Grok alert generation: {e}")

    # ===== FALLBACK: Generate alerts from available data without Grok =====
    if not alerts:
        # Always generate at least one alert from available synthesis
        alert_level = 'CRITICAL' if severity == 'CRITICAL' else ('HIGH' if severity == 'HIGH' else 'MEDIUM')

        # Alert from Vigil Summary (Grok's main analysis)
        if vigil_situation or vigil_approach:
            alerts.append({
                "alert_id": f"alert_{alert_timestamp}_vigil",
                "alert_level": alert_level,
                "alert_type": "vigil_synthesis",
                "title": vigil_situation[:100] if vigil_situation else f"{risk_type} Risk Identified",
                "summary": vigil_situation[:200] if vigil_situation else f"A {severity} severity {risk_type} risk requires attention.",
                "action": vigil_approach[:200] if vigil_approach else "Review full analysis for recommended actions.",
                "source": {"supabase": False, "grok_intelligence": False, "vigil_summary": True}
            })

        # Alert from Grok Intelligence (industry context)
        if grok_context or grok_practices:
            alerts.append({
                "alert_id": f"alert_{alert_timestamp}_grok",
                "alert_level": "MEDIUM" if alert_level == 'CRITICAL' else alert_level,
                "alert_type": "industry_intelligence",
                "title": "Industry Pattern Detected" if grok_context else "Best Practice Available",
                "summary": grok_context[:200] if grok_context else "Industry intelligence available for this risk type.",
                "action": grok_practices[:200] if grok_practices else "Review Grok intelligence for industry context.",
                "source": {"supabase": False, "grok_intelligence": True, "vigil_summary": False}
            })

        # Alert from Supabase patterns
        if supabase_risks:
            similar_count = len(supabase_risks)
            alerts.append({
                "alert_id": f"alert_{alert_timestamp}_supabase",
                "alert_level": "HIGH" if similar_count >= 5 else "MEDIUM",
                "alert_type": "supabase_pattern",
                "title": f"{similar_count} Similar Historical Risks Found",
                "summary": f"Database contains {similar_count} similar {risk_type} risks. Pattern analysis recommended.",
                "action": "Review historical risks for proven mitigation strategies.",
                "source": {"supabase": True, "grok_intelligence": False, "vigil_summary": False}
            })

    return alerts


def _generate_follow_up_questions(
    risk_type: str,
    severity: str,
    vigil_summary: Dict = None,
    grok_intelligence: Dict = None,
    alerts: list = None
) -> list:
    """
    Generate contextual follow-up questions based on the risk analysis.

    Vigil proactively asks follow-up questions when analyzing alerts to help users:
    - Understand the scope and impact of the identified risk
    - Explore industry context and comparisons
    - Investigate historical patterns from their data
    - Guide toward actionable solutions
    """
    questions = []

    if grok_configured:
        try:
            # Build context for question generation
            situation = ""
            if vigil_summary:
                situation = vigil_summary.get('situation') or vigil_summary.get('context', '')

            alert_context = ""
            if alerts:
                alert_titles = [a.get('title', '') for a in alerts[:3]]
                alert_context = "; ".join(alert_titles)

            # Include industry intelligence for more relevant questions
            industry_context = ""
            if grok_intelligence:
                industry_context = grok_intelligence.get('industry_context') or grok_intelligence.get('context', '')

            question_prompt = f"""Based on this risk analysis and alerts, generate 3-5 insightful follow-up questions
that Vigil should ask the user to better understand their situation and provide more targeted help.

Risk Type: {risk_type}
Severity: {severity}
Situation: {situation[:300] if situation else 'Risk identified'}
Alerts Generated: {alert_context if alert_context else 'Standard alerts generated'}
Industry Context: {industry_context[:200] if industry_context else 'General industry analysis'}

Generate questions that:
1. Help clarify the scope and immediate impact on operations
2. Explore industry-specific context and how this compares to similar organizations
3. Investigate whether this matches historical patterns in their data
4. Guide toward actionable solutions with specific next steps
5. Uncover any related risks or dependencies

Make the questions conversational and specific to their situation.
When relevant, reference industry benchmarks or common patterns (you can reference their data in parentheses if applicable).

Return ONLY a JSON array of question objects:
[{{"question": "...", "category": "scope|industry|historical|action|dependencies", "context_hint": "Brief hint about what this explores"}}]"""

            response_text = _call_grok_api(
                prompt=question_prompt,
                system_message="You are Vigil's conversational assistant powered by Grok. Generate insightful follow-up questions that help users understand and address their risks. Be conversational and specific.",
                max_tokens=800,
                temperature=0.5
            )

            if response_text:
                try:
                    questions = json.loads(response_text)
                except json.JSONDecodeError:
                    match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    questions = json.loads(match.group()) if match else []

        except Exception as e:
            logger.warning(f"Follow-up question generation: {e}")

    # Fallback questions if Grok fails
    if not questions:
        questions = [
            {
                "question": f"What's the typical recovery timeline for {risk_type} issues in our industry?",
                "category": "industry",
                "context_hint": "Industry benchmark comparison"
            },
            {
                "question": "Have we seen similar patterns in our historical data?",
                "category": "historical",
                "context_hint": "Pattern analysis from Supabase"
            },
            {
                "question": f"What are the immediate actions we should take for this {severity} severity issue?",
                "category": "action",
                "context_hint": "Prioritized action items"
            },
            {
                "question": "Which departments or stakeholders should be notified?",
                "category": "scope",
                "context_hint": "Impact assessment"
            }
        ]

    return questions


def _generate_solutions(complexity_score: int, risk_type: str) -> list:
    """
    Generate legacy solutions (1-7) for backward compatibility.
    Used when dual-source engine is not available.
    """
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


def _generate_dual_source_solutions(
    transformer,
    risk_type: str,
    severity: str,
    description: str,
    complexity_score: int,
    structured_data: dict = None
) -> dict:
    """
    Generate solutions from BOTH Supabase (private data) and Grok (external intelligence).

    Returns tiered solutions ranked from lowest to highest urgency, with:
    - Source attribution (Private Data vs External Intelligence)
    - Reference links (doc_id for Supabase, URL for Grok)
    - (Recommended) tagging when urgency matches severity

    Tiering:
    - Tier 4: PLANNED (LOW severity)
    - Tier 3: STANDARD (MEDIUM severity)
    - Tier 2: URGENT (HIGH severity)
    - Tier 1: IMMEDIATE (CRITICAL severity)
    """
    if not hasattr(transformer, 'solution_engine') or transformer.solution_engine is None:
        # Fallback to legacy solutions
        legacy = _generate_solutions(complexity_score, risk_type)
        return {
            'has_solutions': len(legacy) > 0,
            'risk_severity': severity,
            'matched_urgency': _severity_to_urgency(severity),
            'tiered_solutions': [{
                'tier': 3,
                'urgency': 'STANDARD',
                'urgency_label': 'Tier 3: STANDARD (Normal Operations)',
                'solutions': [{
                    'title': s['title'],
                    'description': f"Legacy solution for {risk_type} risk",
                    'source': 'system',
                    'source_type': 'System Default',
                    'reference': None,
                    'reference_type': 'none',
                    'urgency': 'STANDARD',
                    'confidence': 0.7,
                    'is_recommended': severity == 'MEDIUM',
                    'success_probability': s['success_probability']
                } for s in legacy]
            }],
            'total_solutions': len(legacy),
            'supabase_count': 0,
            'grok_count': 0
        }

    # Use the dual-source solution engine
    return transformer.solution_engine.generate_dual_source_solutions(
        risk_type=risk_type,
        severity=severity,
        description=description,
        complexity_score=complexity_score,
        structured_data=structured_data
    )


def _severity_to_urgency(severity: str) -> str:
    """Map severity to urgency level."""
    mapping = {
        'CRITICAL': 'IMMEDIATE',
        'HIGH': 'URGENT',
        'MEDIUM': 'STANDARD',
        'LOW': 'PLANNED'
    }
    return mapping.get(severity, 'STANDARD')


def _format_solutions_for_response(dual_source_result: dict) -> dict:
    """
    Format dual-source solutions for API response.

    Each solution includes:
    - Summary: Brief overview of the solution
    - Steps: Step-by-step instructions with actions, details, responsible parties, and durations
    - Expected Outcome: What success looks like
    - Estimated Timeline: How long implementation takes

    Adds (Recommended) suffix to solution titles where urgency matches severity.
    Formats references as clickable links (URLs) or document IDs.
    """
    formatted_tiers = []

    for tier_data in dual_source_result.get('tiered_solutions', []):
        formatted_solutions = []

        for solution in tier_data.get('solutions', []):
            # Add (Recommended) tag to title if applicable
            title = solution['title']
            if solution.get('is_recommended'):
                title = f"{title} (Recommended)"

            # Format reference based on type
            reference_display = None
            if solution.get('reference_type') == 'url':
                reference_display = {
                    'type': 'url',
                    'value': solution.get('reference'),
                    'label': 'External Resource'
                }
            elif solution.get('reference_type') == 'document':
                reference_display = {
                    'type': 'document',
                    'value': solution.get('reference'),
                    'label': 'Internal Document'
                }

            # Format steps for display
            formatted_steps = []
            for step in solution.get('steps', []):
                formatted_steps.append({
                    'step_number': step.get('step', 0),
                    'action': step.get('action', ''),
                    'details': step.get('details', ''),
                    'responsible_party': step.get('responsible_party', 'Team'),
                    'estimated_duration': step.get('duration', 'TBD')
                })

            formatted_solutions.append({
                'title': title,
                'summary': solution.get('summary', ''),
                'description': solution.get('description', ''),
                'steps': formatted_steps,
                'expected_outcome': solution.get('expected_outcome', ''),
                'estimated_timeline': solution.get('estimated_timeline', ''),
                'solution_category': solution.get('solution_category', ''),
                'source': solution.get('source'),
                'source_type': solution.get('source_type'),
                'reference': reference_display,
                'urgency': solution.get('urgency'),
                'confidence': solution.get('confidence', 0),
                'success_probability': solution.get('success_probability', 0),
                'is_recommended': solution.get('is_recommended', False),
                # Include additional details if available (supplier_details, equipment_details, etc.)
                'resource_details': {
                    k: v for k, v in solution.items()
                    if k.endswith('_details')
                } or None
            })

        formatted_tiers.append({
            'tier': tier_data['tier'],
            'urgency': tier_data['urgency'],
            'urgency_label': tier_data['urgency_label'],
            'solution_count': len(formatted_solutions),
            'solutions': formatted_solutions
        })

    # Extract risk context for response
    risk_context = dual_source_result.get('risk_context', {})

    return {
        'has_solutions': dual_source_result.get('has_solutions', False),
        'risk_severity': dual_source_result.get('risk_severity'),
        'matched_urgency': dual_source_result.get('matched_urgency'),
        'recommendation_note': f"Solutions matching {dual_source_result.get('matched_urgency')} urgency are tagged as (Recommended) for {dual_source_result.get('risk_severity')} severity risks",
        'risk_context': {
            'identified_entities': risk_context.get('key_entities', []),
            'solution_focus_areas': risk_context.get('solution_focus_areas', []),
            'applicable_solution_types': risk_context.get('solution_types', [])
        },
        'tiered_solutions': formatted_tiers,
        'summary': {
            'total_solutions': dual_source_result.get('total_solutions', 0),
            'from_private_data': dual_source_result.get('supabase_count', 0),
            'from_external_intelligence': dual_source_result.get('grok_count', 0)
        }
    }


# =========================================================================
# DATA SYNC SERVICE INTEGRATION
# =========================================================================

try:
    from data_sync_service import DataSyncService, register_sync_routes

    # Initialize sync service
    sync_service = DataSyncService()

    # Register sync routes
    register_sync_routes(app, sync_service)

    logger.info("Data Sync Service initialized and routes registered")
except ImportError as e:
    logger.warning(f"Data Sync Service not available: {e}")
    sync_service = None


# =========================================================================
# MAIN
# =========================================================================

if __name__ == '__main__':
    if FLASK_ENV == 'production':
        logger.warning("Use Gunicorn in production!")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)