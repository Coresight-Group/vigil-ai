from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from dualpathtransformer import DualPathRiskTransformer
import os
import json
from datetime import datetime
import requests
from supabase import create_client, Client
import numpy as np
from typing import Dict, List, Optional, Tuple
import uuid

# Import VIGIL utilities and configuration
try:
    from utils import (
        create_synthesis_statement, determine_consensus_level,
        create_synthesis_alert, calculate_urgency_from_severity,
        get_escalation_path, get_notification_channels,
        calculate_combined_confidence, format_synthesis_section,
        validate_problem, validate_alert
    )
except ImportError:
    print("Note: utils.py not found - using inline implementations")

try:
    import config
except ImportError:
    print("Note: config.py not found - using default configuration")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL', 'your_supabase_url')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'your_supabase_key')
XAI_API_KEY = os.getenv('XAI_API_KEY', 'your_xai_api_key')
XAI_API_URL = 'https://api.x.ai/v1/chat/completions'

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    SUPABASE_CONNECTED = True
except Exception as e:
    print(f"Supabase connection error: {e}")
    SUPABASE_CONNECTED = False

# Initialize model
try:
    model = DualPathRiskTransformer()
    model.eval()
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    MODEL_LOADED = False

# Store conversation history
conversation_history = []
alerts_database = []

# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════

SYNTHESIS_SYSTEM_PROMPT = """You are VIGIL, enterprise risk management analyst.

CRITICAL INSTRUCTION - DUAL-SOURCE SYNTHESIS WITH ATTRIBUTION:

You are receiving analysis from TWO SOURCES:
- (Private Source): Your company's actual incident history (Supabase)
- (Vigil): Industry knowledge and best practices (Grok research)

Your task: Synthesize both sources together, always citing where each came from.

FORMAT:

1. EXPLICIT PROBLEM ANSWERED
   - Direct answer to what user asked
   - Clear and concise
   
2. INFERRED PROBLEMS EXPLAINED
   - What data suggests (from both sources)
   - Why it matters
   - Confidence level
   
3. SOLUTIONS OFFERED
   - Your Proven Solutions (Private Source) with success rates
   - Industry Best Practices (Vigil) with benchmarks
   - Hybrid Recommendations (combining both)
   
4. SYNTHESIS STATEMENT
   - How both sources align or differ
   - Consensus level (HIGH/MEDIUM/LOW)
   - Confidence score
   
5. RECOMMENDATIONS
   - Immediate actions (your proven approaches)
   - Short-term (add industry resilience)
   - Long-term (strategic improvements)

CRITICAL RULES:
✓ Always cite source: (Private Source) or (Vigil)
✓ Show when sources agree (highest confidence)
✓ Explain why each source matters
✓ Rank by: Consensus × Effectiveness × Cost-Efficiency
✓ Format based on question type:
  - Pattern questions: Structured Analysis (YES/NO → Evidence → Confidence)
  - Action questions: Prioritized Actions (Immediate → Short → Medium → Long)
  - Comparison questions: Comparison Matrix (Side-by-side → Scoring → Winner)

Remember: Synthesis = Your proven success + Industry wisdom = Best outcome
"""

FORMAT_DETECTION_PROMPT = """Analyze this question and identify its type:

Question: "{question}"

Types:
1. PATTERN_DETECTION: "Is this a pattern?", "Are we seeing a trend?", "Is this accelerating?"
   Format: Structured Analysis (YES/NO → Evidence → Precedent → Confidence → Implications)

2. ACTION_PLAN: "What should we do?", "How do we fix this?", "What are next steps?"
   Format: Prioritized Actions (Immediate → Short → Medium → Long with Owner/Timeline/Cost)

3. COMPARISON: "Compare X to Y?", "Which is worse?", "How does A compare to B?"
   Format: Comparison Matrix (Side-by-side table → Scoring → Differentiators → Winner)

4. IMPACT_ASSESSMENT: "How bad is this?", "What's the impact?", "How serious?"
   Format: Impact Pyramid (Financial → Operational → Strategic with progressively more detail)

5. ROOT_CAUSE_INVESTIGATION: "What happened?", "Why did this occur?", "What caused this?"
   Format: Timeline + Root Cause Narrative (Chronological → Causes → Contributing Factors → Preventability)

6. STRATEGIC_PLANNING: "What's our risk profile?", "Where are we vulnerable?", "Strategic assessment?"
   Format: Strategic Framework (Risk heat map → Categories → Trends → Strategic Implications)

7. GENERAL_ANALYSIS: If none of above clearly fit
   Format: Structured Institutional (Classification → Analysis → Recommendations → Governance)

Respond with ONLY the type name, nothing else. Example: "PATTERN_DETECTION"
"""

# ═══════════════════════════════════════════════════════════════
# DUAL-SOURCE PROBLEM DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_problems_dual_source(
    user_message: str,
    embedding_vector: List[float],
    similar_incidents: List[Dict],
    company_data: Dict
) -> Dict:
    """
    Detect problems from BOTH Supabase (Private Source) and Grok (Vigil).
    Synthesize with attribution.
    """
    
    # Private Source: Detect from Supabase data
    private_source_problems = detect_supabase_problems(
        similar_incidents=similar_incidents,
        company_data=company_data
    )
    
    # Vigil: Get context from Grok
    vigil_problems = query_grok_for_problems(
        user_message=user_message,
        similar_incidents=similar_incidents,
        company_context=company_data,
        private_source_problems=private_source_problems
    )
    
    # Synthesize
    synthesized = synthesize_problems(private_source_problems, vigil_problems)
    
    return {
        'private_source_problems': private_source_problems,
        'vigil_problems': vigil_problems,
        'synthesized_problems': synthesized,
        'consensus_problems': [p for p in synthesized if p.get('consensus') == 'HIGH']
    }


def detect_supabase_problems(similar_incidents: List[Dict], company_data: Dict) -> List[Dict]:
    """Detect problems from your company's incident history (Private Source)"""
    
    problems = []
    
    # Problem 1: Geographic Concentration
    if company_data.get('suppliers'):
        high_concentration = [
            s for s in company_data['suppliers'] 
            if s.get('concentration', 0) > 0.15
        ]
        
        if high_concentration:
            problems.append({
                'type': 'GEOGRAPHIC_CONCENTRATION',
                'problem': f"Geographic concentration exceeds policy ({high_concentration[0]['concentration']:.0%} > 15%)",
                'source': 'Private Source',
                'evidence': [f"{s['name']}: {s['concentration']:.0%}" for s in high_concentration],
                'severity': 'HIGH',
                'confidence': 0.95
            })
    
    # Problem 2: Incident Frequency Acceleration
    if company_data.get('incident_frequency'):
        if company_data['incident_frequency'] > 1.6:
            problems.append({
                'type': 'FREQUENCY_ACCELERATION',
                'problem': f"Incident frequency {company_data['incident_frequency']:.1f}/year exceeds policy 1.6/year",
                'source': 'Private Source',
                'evidence': ['2024 incidents exceeding baseline', 'Acceleration trend +44% YoY'],
                'severity': 'HIGH',
                'confidence': 0.92
            })
    
    return problems


def query_grok_for_problems(
    user_message: str,
    similar_incidents: List[Dict],
    company_context: Dict,
    private_source_problems: List[Dict]
) -> List[Dict]:
    """Ask Grok: What problems does industry knowledge reveal? (Vigil)"""
    
    if not XAI_API_KEY or XAI_API_KEY == 'your_xai_api_key':
        return []
    
    problems_found = []
    
    for ps_problem in private_source_problems:
        
        prompt = f"""
Given this supply chain problem identified from company data:
Problem Type: {ps_problem['type']}
Problem: {ps_problem['problem']}

From industry knowledge, explain:
1. Why this problem is critical
2. What external factors make it worse
3. What industry benchmarks say about this risk
4. Real-world impact (examples from industry)

Be specific about current geopolitical/market context.

Format as JSON with keys: why_critical, external_factors, industry_benchmark, real_world_impact
"""
        
        try:
            response = call_xai_api(prompt, system_content="You are risk assessment expert. Respond ONLY with valid JSON.")
            
            if response:
                problems_found.append({
                    'type': ps_problem['type'],
                    'problem': response.get('why_critical', ps_problem['problem']),
                    'source': 'Vigil',
                    'evidence': response.get('external_factors', []),
                    'severity': ps_problem['severity'],
                    'confidence': 0.85,
                    'context': response
                })
        except:
            pass
    
    return problems_found


def synthesize_problems(private_source: List[Dict], vigil: List[Dict]) -> List[Dict]:
    """Merge problems from both sources with attribution"""
    
    synthesized = []
    used_vigil = set()
    
    for ps_problem in private_source:
        matching_vigil = None
        for i, v_problem in enumerate(vigil):
            if ps_problem['type'] == v_problem['type']:
                matching_vigil = v_problem
                used_vigil.add(i)
                break
        
        if matching_vigil:
            synthesized.append({
                'type': ps_problem['type'],
                'synthesized_statement': (
                    f"{ps_problem['problem']} (Private Source: {ps_problem['evidence'][0]}) - "
                    f"{matching_vigil['problem']} (Vigil: {matching_vigil['evidence'][0] if matching_vigil['evidence'] else 'industry analysis'})"
                ),
                'sources': ['Private Source', 'Vigil'],
                'consensus': 'HIGH',
                'confidence': 0.93,
                'severity': ps_problem['severity'],
                'private_source': ps_problem,
                'vigil': matching_vigil
            })
        else:
            synthesized.append({
                'type': ps_problem['type'],
                'synthesized_statement': ps_problem['problem'],
                'sources': ['Private Source'],
                'consensus': 'UNIQUE',
                'confidence': ps_problem['confidence'],
                'severity': ps_problem['severity'],
                'private_source': ps_problem
            })
    
    return synthesized


# ═══════════════════════════════════════════════════════════════
# ALERT GENERATION & SYNTHESIS
# ═══════════════════════════════════════════════════════════════

def generate_dual_source_alerts(company_data: Dict) -> List[Dict]:
    """Generate alerts from both Private Source and Vigil"""
    
    alerts = []
    
    # Private Source: Rule violations
    private_alerts = []
    
    for supplier in company_data.get('suppliers', []):
        if supplier.get('concentration', 0) > 0.15:
            private_alerts.append({
                'type': 'CONCENTRATION_VIOLATION',
                'severity': 'CRITICAL',
                'source': 'Private Source',
                'data': {
                    'supplier': supplier['name'],
                    'concentration': supplier['concentration'],
                    'policy_limit': 0.15,
                    'violation': supplier['concentration'] - 0.15
                },
                'message': f"{supplier['name']} concentration {supplier['concentration']:.0%} exceeds policy 15%"
            })
    
    # Vigil: Risk context for each private alert
    for alert in private_alerts:
        vigil_context = query_grok_for_alert_context(alert, company_data)
        
        synthesized_alert = {
            'alert_id': str(uuid.uuid4()),
            'type': alert['type'],
            'severity': max(alert['severity'], vigil_context.get('severity', 'HIGH')),
            'urgency': calculate_urgency(alert, vigil_context),
            'synthesis': {
                'statement': f"{alert['message']} (Private Source) - {vigil_context.get('why_it_matters', '')} (Vigil)",
                'sources': ['Private Source', 'Vigil'],
                'consensus': 'HIGH',
                'confidence': 0.95
            },
            'private_source': alert,
            'vigil': vigil_context,
            'recommended_actions': get_recommended_actions(alert, vigil_context),
            'timestamp': datetime.now().isoformat(),
            'status': 'ACTIVE'
        }
        
        alerts.append(synthesized_alert)
    
    return alerts


def query_grok_for_alert_context(alert: Dict, company_data: Dict) -> Dict:
    """Get Vigil context for alert"""
    
    if not XAI_API_KEY or XAI_API_KEY == 'your_xai_api_key':
        return {'why_it_matters': 'Context unavailable', 'severity': 'HIGH'}
    
    prompt = f"""
Alert: {alert['message']}
Company context: Supply chain with {len(company_data.get('suppliers', []))} suppliers across multiple regions.

Explain concisely:
1. Why this matters (from industry perspective)
2. Current geopolitical/market factors making it worse
3. Severity assessment

JSON format: {{why_it_matters: "...", external_factors: [...], severity: "HIGH"}}
"""
    
    try:
        response = call_xai_api(prompt, system_content="You are risk expert. Respond ONLY with JSON.")
        return response if response else {'why_it_matters': alert['message'], 'severity': 'HIGH'}
    except:
        return {'why_it_matters': alert['message'], 'severity': 'HIGH'}


def calculate_urgency(private_alert: Dict, vigil_context: Dict) -> str:
    """Calculate urgency based on severity + consensus"""
    
    if private_alert['severity'] == 'CRITICAL' and vigil_context.get('severity') == 'CRITICAL':
        return 'IMMEDIATE'
    elif private_alert['severity'] == 'CRITICAL':
        return 'URGENT'
    else:
        return 'IMPORTANT'


def get_recommended_actions(private_alert: Dict, vigil_context: Dict) -> List[Dict]:
    """Get recommended actions for alert"""
    
    return [
        {
            'timeframe': 'Immediate',
            'action': 'Review and confirm alert details',
            'owner': 'VP Operations'
        },
        {
            'timeframe': 'Today',
            'action': 'Activate mitigation protocol',
            'owner': 'VP Operations'
        },
        {
            'timeframe': 'This Week',
            'action': 'Implement diversification strategy',
            'owner': 'Procurement'
        }
    ]


# ═══════════════════════════════════════════════════════════════
# FORMAT DETECTION & RESPONSE FORMATTING
# ═══════════════════════════════════════════════════════════════

def detect_question_format(user_message: str) -> str:
    """Detect question type to determine response format"""
    
    if not XAI_API_KEY or XAI_API_KEY == 'your_xai_api_key':
        # Fallback detection
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['pattern', 'trend', 'accelerat']):
            return 'PATTERN_DETECTION'
        elif any(word in message_lower for word in ['should', 'action', 'do we', 'fix']):
            return 'ACTION_PLAN'
        elif any(word in message_lower for word in ['compare', 'versus', 'difference']):
            return 'COMPARISON'
        else:
            return 'GENERAL_ANALYSIS'
    
    try:
        response = call_xai_api(
            FORMAT_DETECTION_PROMPT.format(question=user_message),
            system_content="You are a question classifier. Respond with ONLY the format type name."
        )
        
        if isinstance(response, str):
            return response.strip().upper()
        return 'GENERAL_ANALYSIS'
    except:
        return 'GENERAL_ANALYSIS'


# ═══════════════════════════════════════════════════════════════
# X.AI API INTEGRATION
# ═══════════════════════════════════════════════════════════════

def call_xai_api(
    user_content: str,
    system_content: str = SYNTHESIS_SYSTEM_PROMPT,
    temperature: float = 0.3,
    max_tokens: int = 1500
) -> Optional[Dict]:
    """Call X.AI Grok API with proper error handling"""
    
    if not XAI_API_KEY or XAI_API_KEY == 'your_xai_api_key':
        return None
    
    try:
        headers = {
            'Authorization': f'Bearer {XAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'grok-2',
            'messages': [
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': user_content}
            ],
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        response = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        content = data['choices'][0]['message']['content']
        
        # Try to parse as JSON if user_content asks for it
        try:
            return json.loads(content)
        except:
            return content
    
    except Exception as e:
        print(f"X.AI API error: {e}")
        return None


def get_xai_analysis_formal(
    user_message: str,
    context: str,
    format_type: str = 'GENERAL_ANALYSIS'
) -> str:
    """Get formal X.AI analysis with appropriate format"""
    
    system_prompt = SYNTHESIS_SYSTEM_PROMPT + f"\n\nRESPONSE FORMAT: {format_type}"
    
    response = call_xai_api(
        user_content=context,
        system_content=system_prompt,
        temperature=0.3,
        max_tokens=2000
    )
    
    if isinstance(response, str):
        return response
    
    return str(response)


# ═══════════════════════════════════════════════════════════════
# SUPABASE OPERATIONS
# ═══════════════════════════════════════════════════════════════

def search_supabase_risks(embedding_vector: List[float], top_k: int = 5) -> List[Dict]:
    """Search Supabase for similar risks using pgvector"""
    
    try:
        response = supabase.rpc(
            'match_risks',
            {
                'query_embedding': embedding_vector,
                'match_threshold': 0.7,
                'match_count': top_k
            }
        ).execute()
        
        return response.data if response.data else []
    except Exception as e:
        print(f"Supabase search error: {e}")
        return []


def store_risk_in_supabase(description: str, embedding: List[float], metadata: Dict = None) -> bool:
    """Store risk and embedding in Supabase"""
    
    try:
        data = {
            'description': description,
            'embedding': embedding,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        supabase.table('risks').insert(data).execute()
        return True
    except Exception as e:
        print(f"Supabase store error: {e}")
        return False


def get_company_overview() -> Dict:
    """Get company data for analysis"""
    
    # This would normally come from your database
    # Using synthetic data for now
    return {
        'suppliers': [
            {'name': 'Taiwan TSMC', 'concentration': 0.26, 'region': 'Taiwan', 'critical': True},
            {'name': 'Mexico Assembly', 'concentration': 0.18, 'region': 'Mexico', 'critical': False},
            {'name': 'Malaysia Electronics', 'concentration': 0.22, 'region': 'Malaysia', 'critical': False}
        ],
        'incident_frequency': 2.3,
        'incidents_2024': 5,
        'incidents_2023': 1.6,
        'total_suppliers': 340,
        'regions': 18,
        'policies': {
            'concentration_limit': 0.15,
            'frequency_limit': 1.6,
            'financial_impact_limit': 2000000
        }
    }


# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': MODEL_LOADED,
        'supabase_connected': SUPABASE_CONNECTED,
        'xai_configured': bool(XAI_API_KEY and XAI_API_KEY != 'your_xai_api_key'),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint with dual-source synthesis"""
    
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model not loaded',
            'response': 'System initializing. Please try again.'
        }), 503
    
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message or len(user_message) < 20:
        return jsonify({
            'error': 'invalid_input',
            'response': 'Please provide a detailed risk query (minimum 20 characters).'
        }), 400
    
    try:
        # STEP 1: Embedding
        with torch.no_grad():
            results = model.encode_data(
                data=user_message,
                convert_to_numpy=True,
                store_vectors=True,
                check_alerts=True
            )
        
        embedding_vector = results.get('embeddings', [None])[0]
        if embedding_vector is not None:
            embedding_vector = embedding_vector.tolist()
        
        alerts = results.get('alerts', [])
        
        # STEP 2: Search Supabase
        similar_incidents = []
        if SUPABASE_CONNECTED and embedding_vector:
            similar_incidents = search_supabase_risks(embedding_vector, top_k=10)
        
        # STEP 3: Detect question format
        question_format = detect_question_format(user_message)
        
        # STEP 4: Dual-source problem detection
        company_data = get_company_overview()
        problems = detect_problems_dual_source(
            user_message=user_message,
            embedding_vector=embedding_vector or [],
            similar_incidents=similar_incidents,
            company_data=company_data
        )
        
        # STEP 5: Build context
        context = build_synthesis_context(
            user_message=user_message,
            problems=problems,
            similar_incidents=similar_incidents
        )
        
        # STEP 6: Get X.AI analysis
        xai_response = get_xai_analysis_formal(
            user_message=user_message,
            context=context,
            format_type=question_format
        )
        
        # STEP 7: Store in conversation history
        conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat(),
            'format_detected': question_format
        })
        
        conversation_history.append({
            'role': 'assistant',
            'content': xai_response,
            'timestamp': datetime.now().isoformat(),
            'problems_identified': len(problems['synthesized_problems']),
            'sources_used': ['Private Source', 'Vigil'],
            'consensus_problems': len(problems['consensus_problems'])
        })
        
        # STEP 8: Generate and store alerts
        new_alerts = generate_dual_source_alerts(company_data)
        alerts_database.extend(new_alerts)
        
        return jsonify({
            'response': xai_response,
            'metadata': {
                'format_detected': question_format,
                'problems_identified': len(problems['synthesized_problems']),
                'consensus_problems': len(problems['consensus_problems']),
                'similar_incidents': len(similar_incidents),
                'alerts_generated': len(new_alerts),
                'sources': ['Private Source', 'Vigil']
            },
            'status': 'success'
        }), 200
    
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'error': 'Processing failed',
            'response': 'An error occurred. Please try again.',
            'status': 'error'
        }), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get all active alerts"""
    
    active_alerts = [a for a in alerts_database if a['status'] == 'ACTIVE']
    
    return jsonify({
        'alerts': active_alerts,
        'total_active': len(active_alerts),
        'by_severity': {
            'CRITICAL': len([a for a in active_alerts if a['severity'] == 'CRITICAL']),
            'HIGH': len([a for a in active_alerts if a['severity'] == 'HIGH']),
            'MEDIUM': len([a for a in active_alerts if a['severity'] == 'MEDIUM'])
        },
        'consensus_alerts': len([a for a in active_alerts if a['synthesis']['consensus'] == 'HIGH']),
        'status': 'success'
    }), 200


@app.route('/api/alerts/<alert_id>', methods=['GET'])
def get_alert_details(alert_id: str):
    """Get full alert details with synthesis"""
    
    alert = next((a for a in alerts_database if a['alert_id'] == alert_id), None)
    
    if not alert:
        return jsonify({'error': 'Alert not found'}), 404
    
    return jsonify({
        'alert': alert,
        'status': 'success'
    }), 200


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    
    return jsonify({
        'history': conversation_history,
        'total_messages': len(conversation_history),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    
    global conversation_history
    conversation_history = []
    
    return jsonify({
        'status': 'success',
        'message': 'Conversation history cleared'
    }), 200


def build_synthesis_context(user_message: str, problems: Dict, similar_incidents: List[Dict]) -> str:
    """Build context for Grok with dual-source attribution"""
    
    context = f"""
═════════════════════════════════════════════════════════════════
USER QUERY:
{user_message}

═════════════════════════════════════════════════════════════════
PROBLEMS IDENTIFIED (Dual-Source Synthesis):

"""
    
    for problem in problems['synthesized_problems']:
        if problem.get('sources') and len(problem['sources']) > 1:
            context += f"""
Problem: {problem['type']}
Synthesis: {problem['synthesized_statement']}
Sources: {' + '.join(f"({s})" for s in problem['sources'])}
Consensus: {problem.get('consensus', 'MEDIUM')}
Confidence: {problem.get('confidence', 0.8):.0%}

"""
        else:
            context += f"""
Problem: {problem['type']}
Identified by: {', '.join(f"({s})" for s in problem.get('sources', ['Unknown']))}
Statement: {problem.get('synthesized_statement', problem.get('problem', ''))}
Confidence: {problem.get('confidence', 0.8):.0%}

"""
    
    if similar_incidents:
        context += f"""
═════════════════════════════════════════════════════════════════
SIMILAR HISTORICAL INCIDENTS (Private Source):

"""
        for i, incident in enumerate(similar_incidents[:5], 1):
            context += f"{i}. {incident.get('description', 'N/A')}\n"
    
    context += f"""
═════════════════════════════════════════════════════════════════
PROVIDE COMPREHENSIVE ANALYSIS:
1. Answer explicit question
2. Explain inferred problems
3. Show sources: (Private Source) and (Vigil)
4. Recommend solutions
5. Show confidence levels
═════════════════════════════════════════════════════════════════
"""
    
    return context


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
