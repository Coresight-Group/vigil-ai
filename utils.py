# VIGIL System Utilities & Helper Functions
# Contains reusable functions for synthesis, alerts, and analysis

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# ENUMS FOR TYPE SAFETY
# ═══════════════════════════════════════════════════════════════

class Severity(Enum):
    """Alert severity levels"""
    CRITICAL = 'CRITICAL'
    HIGH = 'HIGH'
    MEDIUM = 'MEDIUM'
    LOW = 'LOW'

class Source(Enum):
    """Data source attribution"""
    PRIVATE_SOURCE = 'Private Source'
    VIGIL = 'Vigil'
    HYBRID = 'Hybrid'

class Consensus(Enum):
    """Problem/Alert consensus level"""
    HIGH = 'HIGH'       # Both sources agree
    MEDIUM = 'MEDIUM'   # Partial agreement
    UNIQUE = 'UNIQUE'   # Only one source

class Urgency(Enum):
    """Alert urgency levels"""
    IMMEDIATE = 'IMMEDIATE'
    URGENT = 'URGENT'
    IMPORTANT = 'IMPORTANT'
    MONITOR = 'MONITOR'

# ═══════════════════════════════════════════════════════════════
# SYNTHESIS UTILITIES
# ═══════════════════════════════════════════════════════════════

def create_synthesis_statement(
    private_source_fact: str,
    vigil_fact: str,
    confidence_private: float,
    confidence_vigil: float
) -> Tuple[str, float]:
    """
    Create a synthesized statement combining both sources.
    
    Returns: (synthesized_statement, combined_confidence)
    """
    
    # Calculate combined confidence (average when both present)
    combined_confidence = (confidence_private + confidence_vigil) / 2
    
    # Create attributed statement
    statement = (
        f"{private_source_fact} (Private Source: {confidence_private:.0%} confidence) "
        f"aligned with {vigil_fact} (Vigil: {confidence_vigil:.0%} confidence). "
        f"Combined assessment: {combined_confidence:.0%} confidence."
    )
    
    return statement, combined_confidence


def determine_consensus_level(
    sources_count: int,
    sources_agree: bool,
    confidence_spread: float
) -> str:
    """
    Determine consensus level based on source agreement.
    
    sources_count: 1 or 2 (one or both sources)
    sources_agree: True if both sources reached same conclusion
    confidence_spread: Difference between source confidences (0-1)
    """
    
    if sources_count == 2 and sources_agree and confidence_spread < 0.2:
        return Consensus.HIGH.value
    elif sources_count == 2 and confidence_spread < 0.3:
        return Consensus.MEDIUM.value
    else:
        return Consensus.UNIQUE.value


def create_attributed_problem(
    problem_type: str,
    private_source_data: Dict,
    vigil_data: Optional[Dict] = None,
    consensus: Optional[str] = None,
    confidence: Optional[float] = None
) -> Dict:
    """
    Create a problem object with full attribution.
    """
    
    problem = {
        'type': problem_type,
        'timestamp': datetime.now().isoformat(),
        'sources': ['Private Source'] + (['Vigil'] if vigil_data else []),
        'consensus': consensus or Consensus.UNIQUE.value,
        'private_source': private_source_data
    }
    
    if vigil_data:
        problem['vigil'] = vigil_data
    
    if confidence is not None:
        problem['confidence'] = confidence
    
    return problem


def create_attributed_solution(
    solution_name: str,
    description: str,
    source: str,
    effectiveness: float,
    timeline_days: int,
    cost: float,
    proof: str,
    **kwargs
) -> Dict:
    """
    Create a solution object with source attribution.
    """
    
    solution = {
        'name': solution_name,
        'description': description,
        'source': source,
        'effectiveness': effectiveness,
        'timeline_days': timeline_days,
        'cost': cost,
        'proof': proof,  # Why we know this works
        'timestamp': datetime.now().isoformat()
    }
    
    # Add any additional kwargs
    solution.update(kwargs)
    
    return solution


# ═══════════════════════════════════════════════════════════════
# ALERT UTILITIES
# ═══════════════════════════════════════════════════════════════

def create_synthesis_alert(
    alert_type: str,
    severity: str,
    private_source_message: str,
    vigil_context: str,
    recommended_actions: List[Dict],
    escalation_path: List[str],
    **kwargs
) -> Dict:
    """Create a fully-attributed alert from both sources."""
    
    import uuid
    
    alert = {
        'alert_id': str(uuid.uuid4()),
        'type': alert_type,
        'severity': severity,
        'urgency': calculate_urgency_from_severity(severity),
        'synthesis': {
            'statement': f"{private_source_message} (Private Source) - {vigil_context} (Vigil)",
            'sources': ['Private Source', 'Vigil'],
            'consensus': Consensus.HIGH.value,
            'confidence': 0.95
        },
        'private_source': {
            'message': private_source_message,
            'source': Source.PRIVATE_SOURCE.value
        },
        'vigil': {
            'context': vigil_context,
            'source': Source.VIGIL.value
        },
        'recommended_actions': recommended_actions,
        'escalation_path': escalation_path,
        'timestamp': datetime.now().isoformat(),
        'status': 'ACTIVE'
    }
    
    # Add any additional kwargs
    alert.update(kwargs)
    
    return alert


def calculate_urgency_from_severity(severity: str) -> str:
    """Map severity to urgency level."""
    
    mapping = {
        'CRITICAL': Urgency.IMMEDIATE.value,
        'HIGH': Urgency.URGENT.value,
        'MEDIUM': Urgency.IMPORTANT.value,
        'LOW': Urgency.MONITOR.value
    }
    
    return mapping.get(severity, Urgency.MONITOR.value)


def get_escalation_path(severity: str) -> List[str]:
    """Get escalation path based on severity."""
    
    paths = {
        'CRITICAL': ['VP_OPERATIONS', 'CFO', 'BOARD'],
        'HIGH': ['VP_OPERATIONS', 'FINANCE'],
        'MEDIUM': ['PROCUREMENT', 'OPERATIONS'],
        'LOW': ['MONITORING_TEAM']
    }
    
    return paths.get(severity, ['MONITORING_TEAM'])


def get_notification_channels(severity: str) -> List[str]:
    """Get notification channels based on severity."""
    
    channels = {
        'CRITICAL': ['SMS', 'EMAIL', 'DASHBOARD', 'SLACK', 'PHONE'],
        'HIGH': ['EMAIL', 'DASHBOARD', 'SLACK'],
        'MEDIUM': ['DASHBOARD', 'EMAIL_DIGEST'],
        'LOW': ['DASHBOARD_WIDGET']
    }
    
    return channels.get(severity, ['DASHBOARD_WIDGET'])


# ═══════════════════════════════════════════════════════════════
# CONFIDENCE & SCORING UTILITIES
# ═══════════════════════════════════════════════════════════════

def calculate_combined_confidence(
    private_confidence: float,
    vigil_confidence: float,
    weight_private: float = 0.5,
    weight_vigil: float = 0.5
) -> float:
    """
    Calculate weighted combined confidence from both sources.
    
    Default: equal weighting (50/50)
    You can adjust weights based on priority
    """
    
    return (private_confidence * weight_private) + (vigil_confidence * weight_vigil)


def score_solution_with_attribution(
    solution: Dict,
    problem: Dict,
    weights: Dict
) -> Tuple[float, Dict]:
    """
    Score a solution based on multiple factors.
    
    Returns: (composite_score, scoring_breakdown)
    """
    
    scores = {
        'effectiveness': solution.get('effectiveness', 0.5) * weights.get('effectiveness', 0.35),
        'applicability': calculate_applicability(solution, problem) * weights.get('applicability', 0.25),
        'cost_efficiency': calculate_cost_score(solution) * weights.get('cost_efficiency', 0.15),
        'timeline': calculate_timeline_score(solution) * weights.get('timeline', 0.15),
        'risk': calculate_risk_score(solution) * weights.get('risk_level', 0.10)
    }
    
    composite = sum(scores.values())
    
    return composite, scores


def calculate_applicability(solution: Dict, problem: Dict) -> float:
    """Calculate how applicable a solution is to a problem (0-1)."""
    
    # Simple heuristic: if solution type matches problem type, high applicability
    if solution.get('type') == problem.get('type'):
        return 0.95
    
    # If from same source, moderate applicability
    if solution.get('source') in problem.get('sources', []):
        return 0.75
    
    # Otherwise, generic applicability
    return 0.5


def calculate_cost_score(solution: Dict) -> float:
    """Normalize cost to 0-1 score (lower cost = higher score)."""
    
    cost = solution.get('cost', 100000)
    
    # Normalize to 0-1: full score at <$100K, zero at $1M+
    if cost >= 1000000:
        return 0.0
    
    return 1.0 - (cost / 1000000)


def calculate_timeline_score(solution: Dict) -> float:
    """Normalize timeline to 0-1 score (shorter timeline = higher score)."""
    
    days = solution.get('timeline_days', 30)
    
    # Normalize to 0-1: full score at <7 days, zero at >180 days
    if days >= 180:
        return 0.0
    
    return 1.0 - (days / 180)


def calculate_risk_score(solution: Dict) -> float:
    """Calculate risk score (lower risk = higher score)."""
    
    risk_level = solution.get('risk_level', 0.3)
    
    # Risk level should be 0-1, invert for scoring
    return 1.0 - risk_level


# ═══════════════════════════════════════════════════════════════
# FORMATTING UTILITIES
# ═══════════════════════════════════════════════════════════════

def format_synthesis_section(title: str, content: str, sources: List[str]) -> str:
    """Format a synthesis section with source attribution."""
    
    source_str = ' + '.join(f"({s})" for s in sources)
    
    return f"""
═════════════════════════════════════════════════════════════════
{title}
Sources: {source_str}
═════════════════════════════════════════════════════════════════

{content}
"""


def format_confidence_statement(confidence: float) -> str:
    """Format confidence as readable statement."""
    
    if confidence >= 0.9:
        return f"VERY HIGH confidence ({confidence:.0%})"
    elif confidence >= 0.8:
        return f"HIGH confidence ({confidence:.0%})"
    elif confidence >= 0.7:
        return f"MODERATE confidence ({confidence:.0%})"
    elif confidence >= 0.6:
        return f"MEDIUM confidence ({confidence:.0%})"
    else:
        return f"LOW confidence ({confidence:.0%})"


def format_severity_with_color(severity: str) -> str:
    """Format severity with visual indication (emoji for accessibility)."""
    
    icons = {
        'CRITICAL': '🔴',
        'HIGH': '🟠',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }
    
    icon = icons.get(severity, '⚪')
    return f"{icon} {severity}"


def create_synthesis_report(
    problems: List[Dict],
    solutions: List[Dict],
    confidence_levels: Dict
) -> str:
    """Create a formatted synthesis report."""
    
    report = """
╔═══════════════════════════════════════════════════════════════╗
║              VIGIL DUAL-SOURCE SYNTHESIS REPORT              ║
╚═══════════════════════════════════════════════════════════════╝

ANALYSIS TIMESTAMP: {timestamp}

"""
    
    report = report.format(timestamp=datetime.now().isoformat())
    
    report += """
─────────────────────────────────────────────────────────────────
PROBLEMS IDENTIFIED (with attribution)
─────────────────────────────────────────────────────────────────

"""
    
    for i, problem in enumerate(problems, 1):
        sources_str = " + ".join(f"({s})" for s in problem.get('sources', ['Unknown']))
        report += f"""
{i}. {problem.get('type', 'Unknown')}
   Statement: {problem.get('synthesized_statement', 'N/A')}
   Sources: {sources_str}
   Consensus: {problem.get('consensus', 'UNKNOWN')}
   Confidence: {problem.get('confidence', 0.0):.0%}

"""
    
    report += """
─────────────────────────────────────────────────────────────────
RECOMMENDED SOLUTIONS (ranked by effectiveness)
─────────────────────────────────────────────────────────────────

"""
    
    for i, solution in enumerate(solutions, 1):
        report += f"""
{i}. {solution.get('name', 'Unknown')}
   Source: ({solution.get('source', 'Unknown')})
   Effectiveness: {solution.get('effectiveness', 0.0):.0%}
   Timeline: {solution.get('timeline_days', 'N/A')} days
   Cost: ${solution.get('cost', 0):,.0f}
   Proof: {solution.get('proof', 'N/A')}

"""
    
    report += """
─────────────────────────────────────────────────────────────────
CONFIDENCE SUMMARY
─────────────────────────────────────────────────────────────────

"""
    
    for key, value in confidence_levels.items():
        report += f"  {key}: {value:.0%}\n"
    
    report += "\n═══════════════════════════════════════════════════════════════\n"
    
    return report


# ═══════════════════════════════════════════════════════════════
# VALIDATION UTILITIES
# ═══════════════════════════════════════════════════════════════

def validate_problem(problem: Dict) -> Tuple[bool, List[str]]:
    """Validate a problem object has required fields."""
    
    required = ['type', 'sources', 'confidence']
    errors = []
    
    for field in required:
        if field not in problem:
            errors.append(f"Required field '{field}' missing")
    
    if problem.get('confidence', 0) < 0 or problem.get('confidence', 0) > 1:
        errors.append("Confidence must be between 0 and 1")
    
    return len(errors) == 0, errors


def validate_solution(solution: Dict) -> Tuple[bool, List[str]]:
    """Validate a solution object has required fields."""
    
    required = ['name', 'source', 'effectiveness', 'timeline_days', 'cost']
    errors = []
    
    for field in required:
        if field not in solution:
            errors.append(f"Required field '{field}' missing")
    
    if solution.get('effectiveness', 0) < 0 or solution.get('effectiveness', 0) > 1:
        errors.append("Effectiveness must be between 0 and 1")
    
    return len(errors) == 0, errors


def validate_alert(alert: Dict) -> Tuple[bool, List[str]]:
    """Validate an alert object has required fields."""
    
    required = ['alert_id', 'type', 'severity', 'synthesis']
    errors = []
    
    for field in required:
        if field not in alert:
            errors.append(f"Required field '{field}' missing")
    
    if alert.get('severity') not in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        errors.append(f"Invalid severity: {alert.get('severity')}")
    
    return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════
# JSON SERIALIZATION FOR API RESPONSES
# ═══════════════════════════════════════════════════════════════

def serialize_for_json(obj):
    """Custom JSON serializer for complex objects."""
    
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    
    return str(obj)


def prepare_api_response(
    data: Dict,
    include_sources: bool = True,
    include_confidence: bool = True,
    include_timestamp: bool = True
) -> Dict:
    """Prepare data for API response with optional fields."""
    
    response = data.copy()
    
    if not include_sources:
        response.pop('sources', None)
    
    if not include_confidence:
        response.pop('confidence', None)
    
    if include_timestamp:
        response['timestamp'] = datetime.now().isoformat()
    
    return response
