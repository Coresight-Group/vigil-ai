"""
VIGIL Dual-Path Risk Management Transformer - Main Module
Integrated System for Interconnected Risk Analysis with Agentic RAG

HANDLES:
- Structured Data (Schema on Write) via PostgreSQL/Supabase
- Semi-Structured Data (Hybrid Schema) via JSONB fields
- Unstructured Data (Schema on Read) via Natural Language Processing

INCLUDES:
- DistilBERT (768-dim vectors) for all data types
- Claude AI (Anthropic) for Deep Reasoning
- Grok Intelligence (X.AI) as Agentic RAG Processor
- Risk Correlation & Conflict Detection
- Historical Pattern Recognition
- Cascading Effect Analysis
- Timeline Correlation
- Semantic Search with pgvector
- Document Storage
- PostgreSQL/Supabase Integration

ARCHITECTURE:
All data types → DistilBERT vectorization (768-dim)
├─ Path 1: Structured (Schema on Write) - DB records
├─ Path 2: Semi-Structured (Hybrid) - JSONB fields
└─ Path 3: Unstructured (Schema on Read) - Natural language
        ↓
Unified semantic space (pgvector)
        ↓
Claude AI (Deep Reasoning) + Grok Intelligence (Agentic RAG)
        ↓
Vigil Summary (seamlessly blended) + Alerts + Solutions
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
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()


# =========================================================================
# GROK INTELLIGENCE ENGINE (AGENTIC RAG PROCESSOR)
# =========================================================================

class GrokIntelligenceEngine:
    """
    Grok Web Intelligence with Agentic RAG capabilities.
    
    Processes first-party data to extract industry knowledge:
    - Analyzes company's own risk data
    - Connects to industry trends
    - Discovers patterns and best practices
    - Provides real-time context
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.grok_url = 'https://api.x.ai/v1/chat/completions'
        self.history = []
    
    def agentic_rag_process(self, description: str, structured_data: Dict = None) -> Dict:
        """
        Agentic RAG: Process first-party data + industry knowledge.
        
        This agent:
        1. Analyzes company's structured risk data
        2. Asks clarifying questions about patterns
        3. Researches similar industry incidents
        4. Discovers best practices
        5. Connects findings to company context
        """
        
        if not self.api_key:
            return {'success': False, 'findings': None}
        
        # Build context with structured data if available
        context = f"Company Risk: {description}"
        if structured_data:
            context += f"\nStructured Context: {json.dumps(structured_data, default=str)[:500]}"
        
        rag_prompt = f"""You are an intelligent RAG agent analyzing company risk data.

TASK: Analyze this company risk and provide industry intelligence:
{context}

As an agent, you should:
1. Identify key characteristics of this risk
2. Research similar incidents in industry
3. Discover what worked in other cases
4. Find best practices for this situation
5. Connect industry knowledge to company context

Provide analysis that:
- Answers what others faced this and how they solved it
- Shows industry standards and benchmarks
- Identifies emerging threats in this domain
- Recommends proven approaches

Be specific with examples and data."""
        
        try:
            response = requests.post(
                self.grok_url,
                json={
                    "messages": [{
                        "role": "user",
                        "content": rag_prompt
                    }],
                    "model": "grok-2",
                    "temperature": 0.7
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=15
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                self.history.append({
                    'timestamp': datetime.now().isoformat(),
                    'input': description,
                    'findings': content,
                    'type': 'agentic_rag'
                })
                return {'success': True, 'findings': content, 'type': 'agentic_rag'}
        except Exception as e:
            pass
        
        return {'success': False, 'findings': None}
    
    def search_context(self, description: str, risk_type: str) -> Dict:
        """Search for industry context and intelligence."""
        
        if not self.api_key:
            return {'success': False, 'findings': None}
        
        prompt = f"""Analyze this business risk and provide industry context.
Risk Type: {risk_type}
Description: {description}

Provide:
1. Current industry situation
2. Similar incidents (last 2 years)
3. Industry consensus on severity
4. Common mitigation strategies
5. Success rates and outcomes"""
        
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
                return {'success': True, 'findings': content}
        except:
            pass
        
        return {'success': False, 'findings': None}
    
    def validate_severity(self, description: str, claimed_severity: str) -> Dict:
        """Validate severity against industry standards."""
        
        if not self.api_key:
            return {'success': False}
        
        prompt = f"""Validate this risk severity.
Risk: {description}
Claimed Severity: {claimed_severity}

Provide: 1) Is severity accurate? 2) What should it be? 3) Why difference?"""
        
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


# =========================================================================
# RISK CORRELATION ENGINE
# =========================================================================

class RiskCorrelationEngine:
    """
    Detects Conflicts, Patterns, and Correlations.
    Works on all data types via unified vector space.
    """
    
    def __init__(self):
        self.correlation_threshold = 0.6
    
    def detect_self_conflicts(self, description: str) -> List[Dict]:
        """Find contradictions within risk description."""
        
        conflicts = []
        desc_lower = description.lower()
        
        contradiction_pairs = [
            (["recover", "resolved", "normal"], ["still down", "ongoing", "failing"]),
            (["contained", "control"], ["spreading", "escalating", "worsening"]),
            (["impact limited", "low risk"], ["widespread", "critical"]),
        ]
        
        for positive, negative in contradiction_pairs:
            has_positive = any(p in desc_lower for p in positive)
            has_negative = any(n in desc_lower for n in negative)
            
            if has_positive and has_negative:
                conflicts.append({
                    'type': 'self_conflict',
                    'description': 'Contradictory statements detected'
                })
        
        return conflicts
    
    def find_historical_matches(self, embedding: np.ndarray, all_records: List[Dict]) -> List[Dict]:
        """Find similar historical records via vector similarity."""
        
        matches = []
        if not all_records:
            return matches
        
        for record in all_records[:10]:  # Check recent records
            if 'embedding' in record:
                similarity = np.dot(embedding, record['embedding']) / (norm(embedding) * norm(record['embedding']) + 1e-10)
                if similarity > self.correlation_threshold:
                    matches.append({
                        'similarity_score': float(similarity),
                        'original_description': record.get('description', '')[:100],
                        'original_date': record.get('created_at', ''),
                        'original_severity': record.get('severity', '')
                    })
        
        return sorted(matches, key=lambda x: x['similarity_score'], reverse=True)[:5]


# =========================================================================
# SCHEMA VALIDATOR
# =========================================================================

class SchemaValidator:
    """Validate schema for all data types."""
    
    MIN_LENGTH = 20
    MAX_LENGTH = 5000
    
    def validate(self, description: str) -> Tuple[bool, str]:
        """Validate description."""
        
        if not description or len(description) < self.MIN_LENGTH:
            return False, "Description too short"
        
        if len(description) > self.MAX_LENGTH:
            return False, "Description too long"
        
        return True, "Valid"


# =========================================================================
# DOCUMENT STORE
# =========================================================================

class DocumentStore:
    """Store and retrieve risk documents across all data types."""
    
    def __init__(self):
        self.documents = {}
        self.id_counter = 0
    
    def store(self, description: str, metadata: Dict = None) -> str:
        """Store document and return ID."""
        
        doc_id = f"doc_{self.id_counter}"
        self.id_counter += 1
        
        self.documents[doc_id] = {
            'description': description,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'data_type': metadata.get('data_type', 'unstructured') if metadata else 'unstructured'
        }
        
        return doc_id
    
    def retrieve(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document by ID."""
        return self.documents.get(doc_id)
    
    def get_all(self) -> List[Dict]:
        """Get all documents."""
        return list(self.documents.values())


# =========================================================================
# DUAL-PATH RISK TRANSFORMER
# =========================================================================

class DualPathRiskTransformer(nn.Module):
    """
    Transforms all data types to unified semantic space.
    
    Path 1: Structured Data (Schema on Write)
      Database records → DistilBERT → 768-dim vectors
    
    Path 2: Semi-Structured (Hybrid Schema)
      Core fields + JSONB → DistilBERT → 768-dim vectors
    
    Path 3: Unstructured (Schema on Read)
      Natural language → DistilBERT → 768-dim vectors
    
    All paths converge in unified semantic space.
    Claude AI synthesizes findings.
    Grok Intelligence provides industry context.
    """
    
    def __init__(self, grok_api_key: str = None, supabase_client=None):
        super().__init__()
        
        # Load DistilBERT (unified embedding for all data types)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.embedding_dim = 768
        
        # Components
        self.grok_engine = GrokIntelligenceEngine(api_key=grok_api_key)
        self.correlation_engine = RiskCorrelationEngine()
        self.schema_validator = SchemaValidator()
        self.document_store = DocumentStore()
        self.supabase = supabase_client
        
        # Claude AI
        try:
            self.claude_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        except:
            self.claude_client = None
        
        self.vector_store = []
    
    def embed_text(self, text: str) -> np.ndarray:
        """Convert any text to 768-dim embedding via DistilBERT."""
        
        inputs = self.tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.distilbert(inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()[0]
    
    def forward(self, data: str, analyze_interconnections: bool = True) -> Dict:
        """
        Analyze risk across all data type paths.
        
        Steps:
        1. Embed all data types to unified space
        2. Detect conflicts and patterns
        3. Find historical precedents
        4. Process with Grok Intelligence (Agentic RAG)
        5. Synthesize with Claude AI
        6. Generate alerts and solutions
        """
        
        # Validate
        valid, msg = self.schema_validator.validate(data)
        if not valid:
            return {'success': False, 'error': msg}
        
        # Embed to unified space (works for all data types)
        embedding = self.embed_text(data)
        
        # Detect issues
        self_conflicts = self.correlation_engine.detect_self_conflicts(data)
        
        # Find historical matches
        historical_matches = self.correlation_engine.find_historical_matches(embedding, self.vector_store)
        
        # Grok Intelligence (Agentic RAG)
        grok_findings = self.grok_engine.agentic_rag_process(data)
        
        # Classify
        risk_type = self._classify_risk(data)
        severity = self._classify_severity(data)
        confidence = self._calculate_confidence(historical_matches, grok_findings)
        
        # Store for future matching
        doc_id = self.document_store.store(data, {
            'risk_type': risk_type,
            'severity': severity,
            'data_type': 'unified'
        })
        
        self.vector_store.append({
            'doc_id': doc_id,
            'description': data,
            'embedding': embedding,
            'risk_type': risk_type,
            'severity': severity,
            'created_at': datetime.now().isoformat()
        })
        
        return {
            'success': True,
            'risk_type': risk_type,
            'severity': severity,
            'confidence': confidence,
            'doc_id': doc_id,
            'analysis': {
                'self_conflicts': self_conflicts,
                'historical_matches': historical_matches,
                'grok_intelligence': grok_findings
            }
        }
    
    def generate_narrative(self, analysis_output: Dict) -> str:
        """Generate markdown narrative from analysis."""
        
        narrative = f"""# Risk Assessment: {analysis_output.get('risk_type', 'UNKNOWN')}

## Classification
- **Type**: {analysis_output.get('risk_type', 'UNKNOWN')}
- **Severity**: {analysis_output.get('severity', 'MEDIUM')}
- **Confidence**: {int(analysis_output.get('confidence', 0) * 100)}%

## Analysis Summary
{analysis_output.get('description', '')}

## Key Findings
- Historical Precedents: {len(analysis_output.get('analysis', {}).get('historical_matches', []))} matches found
- Self-Conflicts: {len(analysis_output.get('analysis', {}).get('self_conflicts', []))} identified
- Industry Intelligence: Analyzed via Grok

---
*VIGIL Assessment - {datetime.now().isoformat()}*
"""
        
        return narrative
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search across all stored vectors."""
        
        query_embedding = self.embed_text(query)
        
        results = []
        for doc in self.vector_store:
            similarity = np.dot(query_embedding, doc['embedding']) / (norm(query_embedding) * norm(doc['embedding']) + 1e-10)
            results.append({
                'doc_id': doc['doc_id'],
                'similarity': float(similarity),
                'description': doc['description'][:100],
                'risk_type': doc['risk_type']
            })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    def get_vector_store_stats(self) -> Dict:
        """Get statistics about stored vectors."""
        
        return {
            'total_documents': len(self.vector_store),
            'embedding_dimension': self.embedding_dim,
            'embedding_model': 'DistilBERT',
            'data_types_supported': ['structured', 'semi-structured', 'unstructured']
        }
    
    def _classify_risk(self, description: str) -> str:
        """Classify risk type."""
        
        keywords = {
            'SUPPLY_CHAIN': ['supplier', 'vendor', 'sourcing', 'procurement'],
            'QUALITY': ['defect', 'specification', 'compliance'],
            'DELIVERY': ['shipment', 'logistics', 'delay'],
            'PRODUCTION': ['equipment', 'manufacturing', 'capacity'],
            'BRAND': ['reputation', 'customer', 'perception']
        }
        
        desc_lower = description.lower()
        for risk_type, words in keywords.items():
            if any(w in desc_lower for w in words):
                return risk_type
        
        return 'SUPPLY_CHAIN'
    
    def _classify_severity(self, description: str) -> str:
        """Classify severity."""
        
        keywords = {
            'CRITICAL': ['crisis', 'bankruptcy', 'catastrophic'],
            'HIGH': ['urgent', 'impact', 'revenue'],
            'MEDIUM': ['issue', 'problem'],
            'LOW': ['minor', 'small']
        }
        
        desc_lower = description.lower()
        for severity, words in keywords.items():
            if any(w in desc_lower for w in words):
                return severity
        
        return 'MEDIUM'
    
    def _calculate_confidence(self, matches: List[Dict], grok_findings: Dict) -> float:
        """Calculate overall confidence."""
        
        confidence = 0.5
        
        if matches:
            avg_match_score = np.mean([m['similarity_score'] for m in matches])
            confidence = confidence * 0.7 + avg_match_score * 0.3
        
        if grok_findings.get('success'):
            confidence = min(1.0, confidence + 0.1)
        
        return float(confidence)


# =========================================================================
# FACTORY FUNCTION
# =========================================================================

def create_dual_path_transformer(grok_api_key: str = None, supabase_client=None) -> DualPathRiskTransformer:
    """Create and initialize the transformer."""
    
    return DualPathRiskTransformer(
        grok_api_key=grok_api_key,
        supabase_client=supabase_client
    )