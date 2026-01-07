"""
VIGIL Dual-Path Risk Management Transformer - Main Module
Integrated System for Interconnected Risk Analysis

INCLUDES:
- Grok Web Intelligence Engine
- Risk Correlation & Conflict Detection
- Historical Pattern Recognition
- Cascading Effect Analysis
- Timeline Correlation
- Semantic Search with Vector Storage
- Schema Validation
- Document Storage
- PostgreSQL/Supabase Integration

ARCHITECTURE:
Path 1: Structured Data (PostgreSQL/Supabase)
Path 2: Unstructured Data (Text/Grok)
+ Full Interconnected Analysis
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import json
import re
from typing import Dict, List, Union, Optional, Tuple
import numpy as np
from numpy.linalg import norm
from datetime import datetime, timedelta
import requests
from difflib import SequenceMatcher


# =========================================================================
# GROK INTELLIGENCE ENGINE
# =========================================================================

class GrokIntelligenceEngine:
    """
    Grok Web Search Integration for Industry Intelligence
    Provides real-time market context and validates risk severity
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.grok_url = 'https://api.x.ai/v1/chat/completions'
        self.history = []
    
    def search_context(self, description: str, risk_type: str) -> Dict:
        """Search for industry context and intelligence"""
        
        if not self.api_key:
            return {'success': False, 'findings': None}
        
        prompt = f"""You are analyzing a business risk. Provide current industry context.
Risk Type: {risk_type}
Risk Description: {description}

Provide:
1. Current industry situation (what's happening now)
2. Similar incidents happening elsewhere
3. Industry consensus on severity
4. Typical duration of this type of issue
5. Common mitigation strategies

Be specific with recent events and data."""
        
        try:
            response = requests.post(
                self.grok_url,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "grok-2",
                    "temperature": 0.3
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                self.history.append({
                    'timestamp': datetime.now().isoformat(),
                    'risk_type': risk_type,
                    'findings': content
                })
                return {'success': True, 'findings': content}
        except Exception as e:
            pass
        
        return {'success': False, 'findings': None}
    
    def validate_severity(self, description: str, claimed_severity: str) -> Dict:
        """Validate severity against current industry situation"""
        
        if not self.api_key:
            return {'success': False}
        
        prompt = f"""Validate this risk severity assessment.
Risk: {description}
Claimed Severity: {claimed_severity}

Provide: 1) Is severity accurate? 2) What should it be? 3) Why difference? 4) Industry precedents 5) Confidence 0-100"""
        
        try:
            response = requests.post(
                self.grok_url,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "grok-2",
                    "temperature": 0.3
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return {'success': True, 'validation': content}
        except:
            pass
        
        return {'success': False}
    
    def find_similar_incidents(self, description: str) -> Dict:
        """Find similar incidents in industry history"""
        
        if not self.api_key:
            return {'success': False}
        
        prompt = f"""Find 3 similar incidents in recent industry history (last 2 years).
Current Incident: {description}

For each: 1) When occurred 2) How resolved 3) Outcomes 4) Lessons learned"""
        
        try:
            response = requests.post(
                self.grok_url,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "grok-2",
                    "temperature": 0.3
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return {'success': True, 'incidents': content}
        except:
            pass
        
        return {'success': False}


# =========================================================================
# RISK CORRELATION ENGINE
# =========================================================================

class RiskCorrelationEngine:
    """
    Detects Conflicts, Patterns, and Correlations Across Risks
    Links current risk to historical data and recurring issues
    """
    
    def __init__(self):
        self.correlation_threshold = 0.6
    
    def detect_self_conflicts(self, description: str) -> List[Dict]:
        """Find contradictions WITHIN the risk description"""
        
        conflicts = []
        desc_lower = description.lower()
        
        contradiction_pairs = [
            (["recover", "resolved", "normal"], ["still down", "ongoing", "failing"]),
            (["contained", "control"], ["spreading", "escalating", "worsening"]),
            (["impact limited", "low risk"], ["widespread", "critical", "catastrophic"]),
            (["temporary"], ["permanent", "long-term"]),
            (["improvement"], ["deterioration", "declining"]),
            (["no losses"], ["losses mounting", "significant losses"]),
        ]
        
        for positive, negative in contradiction_pairs:
            has_positive = any(p in desc_lower for p in positive)
            has_negative = any(n in desc_lower for n in negative)
            
            if has_positive and has_negative:
                conflicts.append({
                    'type': 'self_conflict',
                    'severity': 'high',
                    'description': 'Contradictory statements detected',
                    'positive': [p for p in positive if p in desc_lower],
                    'negative': [n for n in negative if n in desc_lower]
                })
        
        return conflicts
    
    def find_historical_matches(self, description: str, risk_type: str, supabase=None) -> List[Dict]:
        """Find similar risks in Supabase history"""
        
        matches = []
        
        if not supabase:
            return matches
        
        try:
            # Query single risks table instead of separate tables
            response = supabase.table('risks').select(
                'id, description, analysis_metadata, created_at'
            ).limit(50).execute()
            
            for historical in response.data:
                if not historical.get('analysis_metadata'):
                    continue
                
                similarity = self._calculate_similarity(
                    description,
                    historical.get('description', '')
                )
                
                if similarity > self.correlation_threshold:
                    matches.append({
                        'id': historical.get('id'),
                        'original_description': historical.get('description'),
                        'original_severity': historical.get('analysis_metadata', {}).get('severity', 'UNKNOWN'),
                        'original_date': historical.get('created_at'),
                        'similarity_score': similarity
                    })
        except Exception as e:
            print(f"Error querying historical matches: {e}")
        
        return sorted(matches, key=lambda x: x['similarity_score'], reverse=True)[:5]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using SequenceMatcher"""
        matcher = SequenceMatcher(None, text1.lower(), text2.lower())
        return matcher.ratio()
    
    def detect_recurring_patterns(self, risk_type: str, supabase=None) -> List[Dict]:
        """Find patterns in similar risk types"""
        
        patterns = []
        
        if not supabase:
            return patterns
        
        try:
            response = supabase.table('risks').select(
                'analysis_metadata'
            ).limit(20).execute()
            
            severity_counts = {}
            
            for risk in response.data:
                analysis = risk.get('analysis_metadata', {})
                if analysis and 'severity' in analysis:
                    severity = analysis['severity'].upper()
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if len(severity_counts) >= 3:
                top_severity = max(severity_counts, key=severity_counts.get)
                patterns.append({
                    'description': f'{risk_type} - {severity_counts.get(top_severity, 0)} incidents of {top_severity} severity',
                    'severity': top_severity.lower(),
                    'frequency': severity_counts.get(top_severity, 0)
                })
        except Exception as e:
            print(f"Error detecting patterns: {e}")
        
        return patterns
    
    def find_cascading_effects(self, description: str, risk_type: str, supabase=None) -> List[Dict]:
        """Find risks that could be affected by current risk"""
        
        effects = []
        
        if not supabase:
            return effects
        
        try:
            response = supabase.table('risks').select(
                'id, description, analysis_metadata'
            ).limit(20).execute()
            
            for risk in response.data:
                if risk['id'] == description[:10]:  # Skip current
                    continue
                
                similarity = self._calculate_similarity(description, risk.get('description', ''))
                
                if similarity > 0.5:
                    analysis = risk.get('analysis_metadata', {})
                    effects.append({
                        'affected_description': risk['description'],
                        'affected_risk_type': risk_type,
                        'affected_severity': analysis.get('severity', 'UNKNOWN'),
                        'cascade_probability': similarity
                    })
        except Exception as e:
            print(f"Error finding cascading effects: {e}")
        
        return effects[:3]
    
    def find_timeline_correlations(self, description: str, risk_type: str, supabase=None) -> List[Dict]:
        """Find other risks occurring in same time period"""
        
        correlations = []
        
        if not supabase:
            return correlations
        
        try:
            response = supabase.table('risks').select(
                'id, created_at'
            ).limit(100).execute()
            
            now = datetime.now()
            window_days = 14
            
            related_count = 0
            for risk in response.data:
                try:
                    risk_date = datetime.fromisoformat(risk['created_at'].replace('Z', '+00:00'))
                    days_diff = abs((now - risk_date).days)
                    
                    if days_diff <= window_days:
                        related_count += 1
                except:
                    pass
            
            if related_count > 0:
                correlations.append({
                    'related_events_count': related_count,
                    'window_days': window_days,
                    'correlation_strength': min(related_count / 5, 1.0)
                })
        except Exception as e:
            print(f"Error finding timeline correlations: {e}")
        
        return correlations


# =========================================================================
# SCHEMA VALIDATOR
# =========================================================================

class SchemaValidator:
    """Validates risk data against schema"""
    
    def __init__(self):
        self.required_fields = ['description']
        self.max_description_length = 5000
        self.min_description_length = 20
    
    def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        """Validate risk data"""
        
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in data or not data[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate description
        if 'description' in data:
            desc = data['description']
            if len(desc) < self.min_description_length:
                errors.append(f"Description too short (min {self.min_description_length} chars)")
            if len(desc) > self.max_description_length:
                errors.append(f"Description too long (max {self.max_description_length} chars)")
        
        return len(errors) == 0, errors


# =========================================================================
# DOCUMENT STORE
# =========================================================================

class DocumentStore:
    """Manages document storage and retrieval"""
    
    def __init__(self):
        self.documents = {}
        self.counter = 0
    
    def add_document(self, content: str, metadata: Dict = None) -> str:
        """Add document and return ID"""
        
        doc_id = f"doc_{self.counter}"
        self.counter += 1
        
        self.documents[doc_id] = {
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document"""
        return self.documents.get(doc_id)
    
    def list_documents(self) -> List[str]:
        """List all document IDs"""
        return list(self.documents.keys())


# =========================================================================
# DUAL-PATH RISK TRANSFORMER
# =========================================================================

class DualPathRiskTransformer(nn.Module):
    """
    Main transformer combining both paths:
    Path 1: Structured data (Supabase) + correlation analysis
    Path 2: Unstructured data (Grok) + intelligent synthesis
    """
    
    def __init__(
        self,
        embedding_model: str = 'distilbert-base-uncased',
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_categories: int = 5,
        num_severity_levels: int = 4,
        grok_api_key: str = None,
        supabase_client=None
    ):
        super().__init__()
        
        # Core components
        self.tokenizer = DistilBertTokenizer.from_pretrained(embedding_model)
        self.bert_model = DistilBertModel.from_pretrained(embedding_model)
        
        # Classification heads
        self.category_classifier = nn.Linear(embedding_dim, num_categories)
        self.severity_classifier = nn.Linear(embedding_dim, num_severity_levels)
        self.confidence_layer = nn.Linear(embedding_dim, 1)
        
        # Interconnected analysis components
        self.correlator = RiskCorrelationEngine()
        self.grok_engine = GrokIntelligenceEngine(api_key=grok_api_key)
        self.validator = SchemaValidator()
        self.doc_store = DocumentStore()
        
        # Vector storage
        self.vector_store = {}
        self.embedding_index = []
        self.embedding_dim = embedding_dim
        
        # Clients
        self.supabase_client = supabase_client
    
    def forward(
        self,
        data: Union[str, Dict],
        analyze_interconnections: bool = False,
        source: str = 'api'
    ) -> Dict:
        """
        Forward pass with full analysis
        
        Args:
            data: Risk description or dict with 'description' key
            analyze_interconnections: Whether to run full interconnected analysis
            source: Data source ('api', 'attachment', 'email', etc.)
        
        Returns:
            Complete analysis results
        """
        
        # Parse input
        if isinstance(data, dict):
            description = data.get('description', '')
        else:
            description = str(data)
        
        # Validate
        is_valid, errors = self.validator.validate({'description': description})
        
        if not is_valid:
            return {'error': errors, 'success': False}
        
        # Tokenize and get embeddings
        inputs = self.tokenizer(
            description,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
        
        # Store document
        doc_id = self.doc_store.add_document(description, {'source': source})
        
        # Get classifications
        with torch.no_grad():
            category_logits = self.category_classifier(embeddings)
            severity_logits = self.severity_classifier(embeddings)
        
        # Map to labels
        category_labels = ["Product Risk", "Service Risk", "Brand Risk", "Supply Chain", "Quality Control"]
        severity_labels = ["Low", "Medium", "High", "Critical"]
        
        category_idx = category_logits.argmax(dim=-1).item()
        severity_idx = severity_logits.argmax(dim=-1).item()
        
        result = {
            'doc_id': doc_id,
            'embedding': embeddings.cpu().numpy().tolist(),
            'data_source': source,
            'risk_type': category_labels[category_idx],
            'severity': severity_labels[severity_idx],
            'confidence': float(torch.softmax(severity_logits, dim=-1).max()),
            'success': True
        }
        
        # ====================================================================
        # INTERCONNECTED RISK ANALYSIS
        # ====================================================================
        
        if analyze_interconnections:
            analysis = {}
            
            # 1. Self-conflicts
            self_conflicts = self.correlator.detect_self_conflicts(description)
            analysis['self_conflicts'] = self_conflicts
            
            # 2. Historical matches
            historical_matches = self.correlator.find_historical_matches(
                description,
                result['risk_type'],
                self.supabase_client
            )
            analysis['historical_matches'] = historical_matches
            
            # 3. Recurring patterns
            recurring_patterns = self.correlator.detect_recurring_patterns(
                result['risk_type'],
                self.supabase_client
            )
            analysis['recurring_patterns'] = recurring_patterns
            
            # 4. Grok intelligence
            grok_context = self.grok_engine.search_context(description, result['risk_type'])
            grok_validation = self.grok_engine.validate_severity(description, result['severity'])
            grok_similar = self.grok_engine.find_similar_incidents(description)
            
            analysis['grok_intelligence'] = {
                'context': grok_context,
                'validation': grok_validation,
                'similar_incidents': grok_similar
            }
            
            # 5. Cascading effects
            cascading_effects = self.correlator.find_cascading_effects(
                description,
                result['risk_type'],
                self.supabase_client
            )
            analysis['cascading_effects'] = cascading_effects
            
            # 6. Timeline correlations
            timeline_correlations = self.correlator.find_timeline_correlations(
                description,
                result['risk_type'],
                supabase=self.supabase_client
            )
            analysis['timeline_correlations'] = timeline_correlations
            
            # Adjust severity based on Grok if needed
            if grok_validation.get('success'):
                if 'critical' in grok_validation.get('validation', '').lower() and result['severity'] != 'Critical':
                    result['severity'] = 'Critical'
                    analysis['severity_adjusted'] = True
            
            result['analysis'] = analysis
        
        return result
    
    def encode_data(self, data: Union[str, Dict]) -> np.ndarray:
        """Encode data to embedding vector"""
        
        with torch.no_grad():
            output = self.forward(data, analyze_interconnections=False)
        
        embedding = np.array(output['embedding'])
        
        # Store in vector store
        doc_id = output['doc_id']
        self.vector_store[doc_id] = {
            'embedding': embedding,
            'metadata': output
        }
        self.embedding_index.append((doc_id, embedding))
        
        return embedding
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using semantic similarity"""
        
        query_embedding = self.encode_data(query)
        
        results = []
        for doc_id, stored_embedding in self.vector_store.items():
            similarity = np.dot(
                query_embedding,
                stored_embedding['embedding']
            ) / (norm(query_embedding) * norm(stored_embedding['embedding']) + 1e-8)
            
            results.append({
                'doc_id': doc_id,
                'similarity': float(similarity),
                'metadata': stored_embedding['metadata']
            })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    def get_vector_store_stats(self) -> Dict:
        """Get vector store statistics"""
        
        return {
            'total_documents': len(self.vector_store),
            'embedding_dimension': self.embedding_dim,
            'vector_store_size': len(self.embedding_index)
        }
    
    def generate_narrative(self, output: Dict) -> str:
        """Generate narrative explanation of interconnected analysis"""
        
        if not output.get('analysis'):
            return "Analysis unavailable"
        
        analysis = output['analysis']
        description = output.get('description', 'Unknown')
        risk_type = output['risk_type']
        severity = output['severity']
        
        narrative = f"""
# INTERCONNECTED RISK ANALYSIS

## Current Risk
{description} ({risk_type}: {severity.upper()})

## Historical Context
"""
        
        historical = analysis.get('historical_matches', [])
        if historical:
            narrative += f"\n⚠️ Found {len(historical)} similar past incidents:\n"
            for h in historical[:2]:
                narrative += f"- {h['original_description']} ({h['original_severity'].upper()}, {h['original_date'][:10]})\n"
        
        patterns = analysis.get('recurring_patterns', [])
        if patterns:
            narrative += f"\n⚠️ RECURRING ISSUE: {patterns[0]['description']}\n"
        
        cascades = analysis.get('cascading_effects', [])
        if cascades:
            narrative += f"\n## Cascading Effects\n"
            narrative += f"This will likely worsen {len(cascades)} other existing risks:\n"
            for c in cascades[:3]:
                narrative += f"- {c['affected_description']} ({c['affected_risk_type'].upper()}: {c['affected_severity'].upper()})\n"
        
        conflicts = analysis.get('self_conflicts', [])
        if conflicts:
            narrative += f"\n## Internal Conflicts Detected\n"
            narrative += f"⚠️ The description contains {len(conflicts)} contradictions\n"
        
        grok = analysis.get('grok_intelligence', {})
        if grok.get('context', {}).get('success'):
            narrative += f"\n## Industry Intelligence (from Grok)\n"
            findings = grok['context'].get('findings', '')
            narrative += f"{findings[:500]}...\n"
        
        timeline = analysis.get('timeline_correlations', [])
        if timeline:
            narrative += f"\n## Timeline Correlation\n"
            narrative += f"Found {timeline[0].get('related_events_count', 0)} other risks in same time period\n"
        
        return narrative


def create_dual_path_transformer(
    grok_api_key: str = None,
    supabase_client=None,
) -> DualPathRiskTransformer:
    """Factory function to create transformer instance"""
    
    return DualPathRiskTransformer(
        grok_api_key=grok_api_key,
        supabase_client=supabase_client
    )
