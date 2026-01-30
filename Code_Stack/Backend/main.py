"""
VIGIL Dual-Path Risk Management Transformer - Main Module
Integrated System for Interconnected Risk Analysis with Agentic RAG

HANDLES:
- Structured Data (Schema on Write) via PostgreSQL/Supabase
- Semi-Structured Data (Hybrid Schema) via JSONB fields
- Unstructured Data (Schema on Read) via Natural Language Processing

INCLUDES:
- DistilBERT (768-dim vectors) for all data types
- Grok Intelligence (X.AI) for Deep Reasoning & Agentic RAG
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
Grok Intelligence (Deep Reasoning + Agentic RAG)
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
import os
import hashlib
from dotenv import load_dotenv
from supabase import create_client, Client
from cryptography.fernet import Fernet

load_dotenv()


# =========================================================================
# MULTI-TENANT CLIENT MANAGER
# =========================================================================

class ClientManager:
    """
    Manages multi-tenant client connections.

    Uses master database to store client registry and dynamically
    creates connections to individual client Supabase projects.
    """

    def __init__(self):
        # Master database connection (stores clients table)
        self.master_url = os.getenv('MASTER_SUPABASE_URL')
        # Service Role Key (private, bypasses RLS) - for backend operations
        self.master_service_key = os.getenv('MASTER_SUPABASE_SERVICE_KEY')
        # Anon Key (public, respects RLS) - stored for reference
        self.master_anon_key = os.getenv('MASTER_SUPABASE_ANON_KEY')
        self.encryption_key = os.getenv('ENCRYPTION_KEY')

        # Backend uses SERVICE KEY to bypass RLS for full admin access
        if self.master_url and self.master_service_key:
            self.master_client = create_client(self.master_url, self.master_service_key)
        else:
            self.master_client = None

        # Cache for client connections
        self._client_cache: Dict[str, Client] = {}
        self._client_info_cache: Dict[str, Dict] = {}

        # Fernet for encryption/decryption
        if self.encryption_key:
            self.fernet = Fernet(self.encryption_key.encode())
        else:
            self.fernet = None

    def get_client_by_slug(self, slug: str) -> Optional[Client]:
        """
        Get a Supabase client connection for a specific client by slug.
        Returns cached connection if available.
        """
        if slug in self._client_cache:
            return self._client_cache[slug]

        if not self.master_client:
            return None

        try:
            result = self.master_client.table('clients').select(
                'id, name, slug, supabase_url, supabase_anon_key, supabase_service_key, '
                'status, tier, settings'
            ).eq('slug', slug).eq('status', 'active').single().execute()

            if result.data:
                client_info = result.data

                # Decrypt service key if encrypted
                service_key = client_info.get('supabase_service_key')
                if service_key and self.fernet:
                    try:
                        service_key = self.fernet.decrypt(service_key.encode()).decode()
                    except:
                        pass  # Key might not be encrypted

                # Create client connection using SERVICE KEY (bypasses RLS)
                client = create_client(
                    client_info['supabase_url'],
                    service_key or client_info['supabase_anon_key']
                )

                # Cache it
                self._client_cache[slug] = client
                self._client_info_cache[slug] = client_info

                # Update last_accessed_at
                self.master_client.table('clients').update({
                    'last_accessed_at': datetime.now().isoformat()
                }).eq('id', client_info['id']).execute()

                return client

        except Exception as e:
            print(f"Error getting client {slug}: {e}")

        return None

    def get_client_by_api_key(self, api_key: str) -> Tuple[Optional[Client], Optional[Dict]]:
        """
        Authenticate and get client connection using API key.
        Returns (client, client_info) or (None, None) if invalid.
        """
        if not self.master_client or not api_key:
            return None, None

        # Extract prefix (first 8 chars after 'vgl_')
        if api_key.startswith('vgl_'):
            key_prefix = api_key[:12]  # 'vgl_' + 8 chars
        else:
            key_prefix = api_key[:8]

        # Hash the full key for comparison
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        try:
            # Find API key and join with client
            result = self.master_client.table('api_keys').select(
                '*, clients(*)'
            ).eq('key_prefix', key_prefix).eq('status', 'active').single().execute()

            if result.data:
                stored_hash = result.data.get('key_hash')
                if stored_hash == key_hash:
                    client_info = result.data.get('clients')
                    if client_info and client_info.get('status') == 'active':
                        # Update API key usage
                        self.master_client.table('api_keys').update({
                            'last_used_at': datetime.now().isoformat(),
                            'usage_count': result.data.get('usage_count', 0) + 1
                        }).eq('id', result.data['id']).execute()

                        # Get or create client connection
                        client = self.get_client_by_slug(client_info['slug'])
                        return client, {
                            **client_info,
                            'scopes': result.data.get('scopes', [])
                        }
        except Exception as e:
            print(f"Error authenticating API key: {e}")

        return None, None

    def get_client_info(self, slug: str) -> Optional[Dict]:
        """Get cached client info without creating connection."""
        if slug in self._client_info_cache:
            return self._client_info_cache[slug]

        # Trigger connection to populate cache
        self.get_client_by_slug(slug)
        return self._client_info_cache.get(slug)

    def list_clients(self, status: str = 'active') -> List[Dict]:
        """List all clients with given status."""
        if not self.master_client:
            return []

        try:
            result = self.master_client.table('clients').select(
                'id, name, slug, status, tier, contact_email, created_at'
            ).eq('status', status).order('name').execute()

            return result.data or []
        except Exception as e:
            print(f"Error listing clients: {e}")
            return []

    def create_client(
        self,
        name: str,
        slug: str,
        supabase_url: str,
        supabase_anon_key: str,
        supabase_service_key: str = None,
        tier: str = 'standard',
        contact_email: str = None,
        settings: Dict = None
    ) -> Optional[Dict]:
        """Register a new client in the master database."""
        if not self.master_client:
            return None

        # Encrypt service key if provided
        encrypted_service_key = None
        if supabase_service_key and self.fernet:
            encrypted_service_key = self.fernet.encrypt(
                supabase_service_key.encode()
            ).decode()
        elif supabase_service_key:
            encrypted_service_key = supabase_service_key

        try:
            result = self.master_client.table('clients').insert({
                'name': name,
                'slug': slug,
                'supabase_url': supabase_url,
                'supabase_anon_key': supabase_anon_key,
                'supabase_service_key': encrypted_service_key,
                'tier': tier,
                'contact_email': contact_email,
                'settings': settings or {},
                'status': 'active'
            }).execute()

            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error creating client: {e}")
            return None

    def generate_api_key(
        self,
        client_id: str,
        name: str = 'Default Key',
        scopes: List[str] = None
    ) -> Optional[Tuple[str, Dict]]:
        """
        Generate a new API key for a client.
        Returns (plain_key, key_record) - plain_key is only shown once!
        """
        if not self.master_client:
            return None

        import secrets

        # Generate key: vgl_<env>_<random>
        env = os.getenv('ENVIRONMENT', 'dev')[:4]
        random_part = secrets.token_urlsafe(32)
        plain_key = f"vgl_{env}_{random_part}"

        key_prefix = plain_key[:12]
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        try:
            result = self.master_client.table('api_keys').insert({
                'client_id': client_id,
                'name': name,
                'key_hash': key_hash,
                'key_prefix': key_prefix,
                'scopes': scopes or ['read:risks', 'write:risks', 'read:documents'],
                'status': 'active'
            }).execute()

            if result.data:
                return plain_key, result.data[0]
        except Exception as e:
            print(f"Error generating API key: {e}")

        return None

    def get_data_sources(self, client_id: str) -> List[Dict]:
        """Get all data sources for a client."""
        if not self.master_client:
            return []

        try:
            result = self.master_client.table('data_sources').select('*').eq(
                'client_id', client_id
            ).order('name').execute()

            return result.data or []
        except Exception as e:
            print(f"Error getting data sources: {e}")
            return []

    def record_usage(
        self,
        client_id: str,
        risks_created: int = 0,
        risks_analyzed: int = 0,
        documents_uploaded: int = 0,
        documents_processed: int = 0,
        api_calls: int = 0,
        grok_tokens: int = 0,
        embeddings_generated: int = 0
    ):
        """Record usage metrics for billing/limits."""
        if not self.master_client:
            return

        today = datetime.now().date()
        period_start = today.replace(day=1)

        # Calculate period end (last day of month)
        if today.month == 12:
            period_end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            period_end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)

        try:
            # Try to update existing record
            existing = self.master_client.table('usage_metrics').select('id').eq(
                'client_id', client_id
            ).eq('period_start', period_start.isoformat()).single().execute()

            if existing.data:
                # Update existing
                self.master_client.rpc('increment_usage_metrics', {
                    'p_id': existing.data['id'],
                    'p_risks_created': risks_created,
                    'p_risks_analyzed': risks_analyzed,
                    'p_documents_uploaded': documents_uploaded,
                    'p_documents_processed': documents_processed,
                    'p_api_calls': api_calls,
                    'p_grok_tokens': grok_tokens,
                    'p_embeddings': embeddings_generated
                }).execute()
            else:
                # Insert new
                self.master_client.table('usage_metrics').insert({
                    'client_id': client_id,
                    'period_start': period_start.isoformat(),
                    'period_end': period_end.isoformat(),
                    'risks_created': risks_created,
                    'risks_analyzed': risks_analyzed,
                    'documents_uploaded': documents_uploaded,
                    'documents_processed': documents_processed,
                    'api_calls': api_calls,
                    'grok_tokens_used': grok_tokens,
                    'embeddings_generated': embeddings_generated
                }).execute()
        except Exception as e:
            # Silently fail - usage tracking shouldn't break operations
            pass


# Global client manager instance
client_manager = ClientManager()


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
# DUAL-SOURCE SOLUTION ENGINE
# =========================================================================

class DualSourceSolutionEngine:
    """
    Generates solutions from BOTH Supabase (private data) and Grok (external intelligence)
    SIMULTANEOUSLY. Both sources always run together - never as fallbacks.

    Source Attribution:
    - Supabase returns: doc_id references (internal documents/attachments)
    - Grok returns: URL links to external resources

    Risk-Type-Specific Solution Categories:
    - SUPPLY_CHAIN: raw_materials, components, logistics
    - QUALITY: testing_equipment, quality_tools, certifications
    - DELIVERY: logistics, packaging, transportation
    - PRODUCTION: equipment, machinery, spare_parts, maintenance
    - BRAND: marketing, pr_services, communications

    Tiering (lowest to highest urgency):
    - Tier 4: PLANNED (LOW severity) - Long-term strategic solutions
    - Tier 3: STANDARD (MEDIUM severity) - Standard operational solutions
    - Tier 2: URGENT (HIGH severity) - Priority solutions requiring attention
    - Tier 1: IMMEDIATE (CRITICAL severity) - Crisis-level solutions

    Recommendation Logic:
    - If solution.urgency == risk.severity, tag as "(Recommended)"
    """

    # Risk type to solution categories mapping
    RISK_TYPE_SOLUTIONS = {
        'SUPPLY_CHAIN': {
            'categories': ['raw_materials', 'components', 'logistics'],
            'solution_types': ['supplier_engagement', 'inventory_reallocation', 'alternative_sourcing', 'contract_renegotiation'],
            'supabase_tables': ['suppliers', 'inventory_items', 'contracts', 'logistics_providers']
        },
        'QUALITY': {
            'categories': ['testing_equipment', 'quality_tools', 'certifications'],
            'solution_types': ['quality_audit', 'certification_review', 'testing_protocol', 'compliance_action'],
            'supabase_tables': ['quality_standards', 'certifications', 'testing_equipment', 'audit_history']
        },
        'DELIVERY': {
            'categories': ['logistics', 'packaging', 'transportation'],
            'solution_types': ['route_optimization', 'carrier_switch', 'packaging_redesign', 'delivery_rescheduling'],
            'supabase_tables': ['logistics_providers', 'carriers', 'shipping_routes', 'packaging_options']
        },
        'PRODUCTION': {
            'categories': ['equipment', 'machinery', 'spare_parts', 'maintenance'],
            'solution_types': ['equipment_repair', 'maintenance_schedule', 'capacity_adjustment', 'workflow_optimization'],
            'supabase_tables': ['equipment', 'spare_parts', 'maintenance_schedules', 'production_lines']
        },
        'BRAND': {
            'categories': ['marketing', 'pr_services', 'communications'],
            'solution_types': ['crisis_communication', 'pr_campaign', 'stakeholder_outreach', 'reputation_recovery'],
            'supabase_tables': ['pr_agencies', 'communication_templates', 'stakeholder_contacts', 'media_contacts']
        }
    }

    # Urgency tiers mapped to severity levels
    URGENCY_TO_SEVERITY = {
        'IMMEDIATE': 'CRITICAL',
        'URGENT': 'HIGH',
        'STANDARD': 'MEDIUM',
        'PLANNED': 'LOW'
    }

    SEVERITY_TO_URGENCY = {
        'CRITICAL': 'IMMEDIATE',
        'HIGH': 'URGENT',
        'MEDIUM': 'STANDARD',
        'LOW': 'PLANNED'
    }

    # Tier ordering (1 = highest urgency, 4 = lowest)
    URGENCY_TIER_ORDER = {
        'IMMEDIATE': 1,
        'URGENT': 2,
        'STANDARD': 3,
        'PLANNED': 4
    }

    def __init__(self, supabase_client=None, grok_engine: GrokIntelligenceEngine = None):
        self.supabase = supabase_client
        self.grok_engine = grok_engine

    def generate_dual_source_solutions(
        self,
        risk_type: str,
        severity: str,
        description: str,
        complexity_score: int = 5,
        structured_data: Dict = None
    ) -> Dict:
        """
        Generate dynamic solutions from both Supabase and Grok simultaneously.
        Each solution includes a summary and step-by-step instructions.

        Returns:
        {
            'has_solutions': bool,
            'risk_severity': str,
            'matched_urgency': str,
            'risk_context': {
                'description': str,
                'key_entities': list,
                'solution_focus_areas': list
            },
            'tiered_solutions': [
                {
                    'tier': 1-4,
                    'urgency': 'IMMEDIATE'|'URGENT'|'STANDARD'|'PLANNED',
                    'solutions': [
                        {
                            'title': str,
                            'summary': str,  # Brief overview of the solution
                            'description': str,
                            'steps': [  # Step-by-step instructions
                                {'step': 1, 'action': str, 'details': str, 'responsible_party': str},
                                ...
                            ],
                            'expected_outcome': str,
                            'estimated_timeline': str,
                            'source': 'supabase'|'grok',
                            'source_type': 'Private Data'|'External Intelligence',
                            'reference': str (doc_id or URL),
                            'reference_type': 'document'|'url',
                            'confidence': float,
                            'is_recommended': bool,
                            'success_probability': float,
                            'solution_category': str
                        }
                    ]
                }
            ],
            'total_solutions': int,
            'supabase_count': int,
            'grok_count': int
        }
        """
        matched_urgency = self.SEVERITY_TO_URGENCY.get(severity, 'STANDARD')

        # Extract key context from the risk description for dynamic solutions
        risk_context = self._analyze_risk_context(description, risk_type, structured_data)

        # Query both sources simultaneously with context
        supabase_solutions = self._query_supabase_solutions(
            risk_type, severity, description, structured_data
        )
        grok_solutions = self._query_grok_solutions(
            risk_type, severity, description, complexity_score
        )

        # Combine all solutions
        all_solutions = supabase_solutions + grok_solutions

        # Enrich solutions with dynamic summaries and steps based on context
        enriched_solutions = self._enrich_solutions_with_details(
            all_solutions, risk_context, severity, complexity_score
        )

        # Tag recommendations where urgency matches severity
        for solution in enriched_solutions:
            solution['is_recommended'] = (
                self.URGENCY_TO_SEVERITY.get(solution['urgency']) == severity
            )

        # Group by tier
        tiered_solutions = self._organize_by_tier(enriched_solutions)

        return {
            'has_solutions': len(enriched_solutions) > 0,
            'risk_severity': severity,
            'matched_urgency': matched_urgency,
            'risk_context': risk_context,
            'tiered_solutions': tiered_solutions,
            'total_solutions': len(enriched_solutions),
            'supabase_count': len(supabase_solutions),
            'grok_count': len(grok_solutions)
        }

    def _analyze_risk_context(
        self,
        description: str,
        risk_type: str,
        structured_data: Dict = None
    ) -> Dict:
        """
        Analyze the risk description to extract key context for dynamic solutions.
        Identifies entities, focus areas, and solution priorities.
        """
        # Extract key entities from description (suppliers, products, locations, etc.)
        key_entities = []
        focus_areas = []

        # Common entity patterns
        description_lower = description.lower()

        # Extract supplier/vendor names
        if structured_data and structured_data.get('supplier'):
            key_entities.append({'type': 'supplier', 'value': structured_data['supplier']})
        if structured_data and structured_data.get('vendor'):
            key_entities.append({'type': 'vendor', 'value': structured_data['vendor']})

        # Extract product/component references
        if structured_data and structured_data.get('product'):
            key_entities.append({'type': 'product', 'value': structured_data['product']})
        if structured_data and structured_data.get('component'):
            key_entities.append({'type': 'component', 'value': structured_data['component']})

        # Extract location references
        if structured_data and structured_data.get('location'):
            key_entities.append({'type': 'location', 'value': structured_data['location']})
        if structured_data and structured_data.get('region'):
            key_entities.append({'type': 'region', 'value': structured_data['region']})

        # Determine solution focus areas based on risk type and keywords
        risk_config = self.RISK_TYPE_SOLUTIONS.get(risk_type, self.RISK_TYPE_SOLUTIONS['SUPPLY_CHAIN'])

        # Dynamic focus area detection
        if 'delay' in description_lower or 'late' in description_lower:
            focus_areas.append('timeline_recovery')
        if 'cost' in description_lower or 'price' in description_lower or 'budget' in description_lower:
            focus_areas.append('cost_mitigation')
        if 'quality' in description_lower or 'defect' in description_lower:
            focus_areas.append('quality_improvement')
        if 'shortage' in description_lower or 'stockout' in description_lower:
            focus_areas.append('inventory_management')
        if 'failure' in description_lower or 'breakdown' in description_lower:
            focus_areas.append('equipment_recovery')
        if 'reputation' in description_lower or 'brand' in description_lower or 'public' in description_lower:
            focus_areas.append('reputation_management')
        if 'compliance' in description_lower or 'regulation' in description_lower:
            focus_areas.append('regulatory_compliance')
        if 'customer' in description_lower or 'client' in description_lower:
            focus_areas.append('customer_relations')

        # Default focus areas from risk type if none detected
        if not focus_areas:
            focus_areas = risk_config['categories'][:2]

        return {
            'description': description,
            'risk_type': risk_type,
            'key_entities': key_entities,
            'solution_focus_areas': focus_areas,
            'solution_types': risk_config['solution_types'],
            'primary_tables': risk_config['supabase_tables']
        }

    def _enrich_solutions_with_details(
        self,
        solutions: List[Dict],
        risk_context: Dict,
        severity: str,
        complexity_score: int
    ) -> List[Dict]:
        """
        Enrich each solution with dynamic summary and step-by-step instructions
        based on the risk context.

        If a solution already has steps (e.g., from Grok API), those are preserved
        and contextualized. Otherwise, steps are generated from templates.
        """
        enriched = []

        for solution in solutions:
            # Generate or enhance summary
            existing_summary = solution.get('summary', '')
            if existing_summary:
                # Enhance existing summary with context
                summary = self._enhance_summary_with_context(existing_summary, solution, risk_context)
            else:
                # Generate new summary
                summary = self._generate_solution_summary(solution, risk_context)

            # Handle steps - use existing if provided, otherwise generate
            existing_steps = solution.get('steps', [])
            if existing_steps and len(existing_steps) > 0:
                # Contextualize existing steps
                steps = self._contextualize_existing_steps(existing_steps, risk_context, severity)
            else:
                # Generate new steps from templates
                steps = self._generate_solution_steps(solution, risk_context, severity, complexity_score)

            # Generate or use existing expected outcome
            existing_outcome = solution.get('expected_outcome', '')
            if existing_outcome:
                expected_outcome = existing_outcome
            else:
                expected_outcome = self._generate_expected_outcome(solution, risk_context)

            # Estimate timeline
            timeline = self._estimate_timeline(solution, severity, complexity_score)

            enriched_solution = {
                **solution,
                'summary': summary,
                'steps': steps,
                'expected_outcome': expected_outcome,
                'estimated_timeline': timeline
            }
            enriched.append(enriched_solution)

        return enriched

    def _enhance_summary_with_context(self, existing_summary: str, solution: Dict, risk_context: Dict) -> str:
        """Enhance an existing summary with risk context."""
        entities = risk_context.get('key_entities', [])
        if entities:
            entity_refs = ', '.join([e['value'] for e in entities[:2]])
            return f"{existing_summary} (Targeting: {entity_refs})"
        return existing_summary

    def _contextualize_existing_steps(
        self,
        steps: List[Dict],
        risk_context: Dict,
        severity: str
    ) -> List[Dict]:
        """Contextualize existing steps with risk-specific details."""
        contextualized = []
        for step in steps:
            ctx_step = {
                'step': step.get('step', len(contextualized) + 1),
                'action': step.get('action', ''),
                'details': self._contextualize_step(step.get('details', ''), risk_context),
                'responsible_party': step.get('responsible_party', 'Team'),
                'duration': self._adjust_duration_for_severity(step.get('duration', '1 day'), severity)
            }
            contextualized.append(ctx_step)
        return contextualized

    def _generate_solution_summary(self, solution: Dict, risk_context: Dict) -> str:
        """Generate a concise summary for the solution based on context."""
        title = solution.get('title', 'Solution')
        category = solution.get('solution_category', 'general')
        description = solution.get('description', '')

        # Build dynamic summary based on focus areas
        focus_areas = risk_context.get('solution_focus_areas', [])
        entities = risk_context.get('key_entities', [])

        # Create contextual summary
        entity_refs = ', '.join([e['value'] for e in entities[:2]]) if entities else 'affected resources'

        if 'timeline_recovery' in focus_areas:
            return f"{title}: Addresses delivery timeline concerns for {entity_refs}. {description[:100]}..."
        elif 'cost_mitigation' in focus_areas:
            return f"{title}: Cost-effective approach to mitigate financial impact on {entity_refs}. {description[:100]}..."
        elif 'quality_improvement' in focus_areas:
            return f"{title}: Quality-focused solution targeting defect reduction for {entity_refs}. {description[:100]}..."
        elif 'inventory_management' in focus_areas:
            return f"{title}: Inventory optimization strategy for {entity_refs}. {description[:100]}..."
        elif 'equipment_recovery' in focus_areas:
            return f"{title}: Equipment restoration plan for {entity_refs}. {description[:100]}..."
        elif 'reputation_management' in focus_areas:
            return f"{title}: Brand protection and stakeholder communication strategy. {description[:100]}..."
        else:
            return f"{title}: {description[:150]}..." if len(description) > 150 else f"{title}: {description}"

    def _generate_solution_steps(
        self,
        solution: Dict,
        risk_context: Dict,
        severity: str,
        complexity_score: int
    ) -> List[Dict]:
        """
        Generate step-by-step instructions dynamically based on solution type and context.
        More steps for higher complexity, faster steps for higher severity.
        """
        category = solution.get('solution_category', 'general')
        urgency = solution.get('urgency', 'STANDARD')
        source = solution.get('source', 'system')

        # Base step templates by category
        step_templates = self._get_step_templates(category, risk_context)

        # Adjust number of steps based on complexity
        if complexity_score <= 3:
            num_steps = min(3, len(step_templates))
        elif complexity_score <= 6:
            num_steps = min(5, len(step_templates))
        else:
            num_steps = min(7, len(step_templates))

        # Build steps with context
        steps = []
        for i, template in enumerate(step_templates[:num_steps]):
            step = {
                'step': i + 1,
                'action': template['action'],
                'details': self._contextualize_step(template['details'], risk_context),
                'responsible_party': template.get('responsible_party', 'Operations Team'),
                'duration': self._adjust_duration_for_severity(template.get('duration', '1 day'), severity)
            }
            steps.append(step)

        return steps

    def _get_step_templates(self, category: str, risk_context: Dict) -> List[Dict]:
        """Get step templates based on solution category."""
        templates = {
            # Supply Chain categories
            'supplier_engagement': [
                {'action': 'Assess Current Supplier Status', 'details': 'Review supplier performance metrics and current contract terms for {entities}', 'responsible_party': 'Procurement', 'duration': '4 hours'},
                {'action': 'Initiate Supplier Communication', 'details': 'Contact supplier representatives to discuss the risk situation and request status update', 'responsible_party': 'Procurement Lead', 'duration': '2 hours'},
                {'action': 'Negotiate Interim Terms', 'details': 'Propose temporary adjustments to delivery schedules or quantities', 'responsible_party': 'Procurement Manager', 'duration': '1 day'},
                {'action': 'Document Agreements', 'details': 'Formalize any changes in writing with updated delivery commitments', 'responsible_party': 'Legal/Procurement', 'duration': '4 hours'},
                {'action': 'Establish Monitoring Protocol', 'details': 'Set up daily/weekly check-ins to track supplier progress', 'responsible_party': 'Supply Chain Analyst', 'duration': '2 hours'}
            ],
            'inventory_reallocation': [
                {'action': 'Audit Current Inventory', 'details': 'Review inventory levels across all locations for {entities}', 'responsible_party': 'Inventory Manager', 'duration': '4 hours'},
                {'action': 'Identify Reallocation Opportunities', 'details': 'Map surplus inventory at other locations that can be redirected', 'responsible_party': 'Supply Chain Analyst', 'duration': '2 hours'},
                {'action': 'Coordinate Transfer Logistics', 'details': 'Arrange transportation for inventory movement between facilities', 'responsible_party': 'Logistics Coordinator', 'duration': '1 day'},
                {'action': 'Update Inventory Systems', 'details': 'Reflect inventory transfers in ERP/WMS systems', 'responsible_party': 'IT/Operations', 'duration': '2 hours'},
                {'action': 'Verify Receipt and Quality', 'details': 'Confirm transferred inventory meets quality standards upon arrival', 'responsible_party': 'Quality Control', 'duration': '4 hours'}
            ],
            'alternative_sourcing': [
                {'action': 'Identify Alternative Suppliers', 'details': 'Research and shortlist qualified alternative suppliers for {entities}', 'responsible_party': 'Procurement', 'duration': '1 day'},
                {'action': 'Request Quotes', 'details': 'Send RFQ to potential suppliers with urgent timeline requirements', 'responsible_party': 'Procurement Analyst', 'duration': '4 hours'},
                {'action': 'Evaluate Supplier Capabilities', 'details': 'Assess quality certifications, capacity, and lead times', 'responsible_party': 'Quality/Procurement', 'duration': '1 day'},
                {'action': 'Negotiate Emergency Terms', 'details': 'Secure favorable pricing for expedited orders', 'responsible_party': 'Procurement Manager', 'duration': '4 hours'},
                {'action': 'Onboard New Supplier', 'details': 'Complete vendor setup in procurement systems', 'responsible_party': 'Procurement/IT', 'duration': '2 hours'},
                {'action': 'Place Initial Order', 'details': 'Submit purchase order with confirmed delivery dates', 'responsible_party': 'Procurement', 'duration': '1 hour'}
            ],
            'contract_renegotiation': [
                {'action': 'Review Current Contract Terms', 'details': 'Analyze existing contract clauses related to the risk situation', 'responsible_party': 'Legal', 'duration': '4 hours'},
                {'action': 'Identify Leverage Points', 'details': 'Document performance issues or force majeure conditions', 'responsible_party': 'Procurement/Legal', 'duration': '2 hours'},
                {'action': 'Draft Amendment Proposal', 'details': 'Prepare contract modification request with specific terms', 'responsible_party': 'Legal', 'duration': '1 day'},
                {'action': 'Schedule Negotiation Meeting', 'details': 'Arrange meeting with supplier stakeholders', 'responsible_party': 'Procurement Manager', 'duration': '2 hours'},
                {'action': 'Execute Amendments', 'details': 'Finalize and sign updated contract terms', 'responsible_party': 'Legal/Executive', 'duration': '1 day'}
            ],
            # Quality categories
            'quality_audit': [
                {'action': 'Define Audit Scope', 'details': 'Identify specific quality areas to audit based on {entities}', 'responsible_party': 'Quality Manager', 'duration': '2 hours'},
                {'action': 'Assemble Audit Team', 'details': 'Assign qualified auditors with relevant expertise', 'responsible_party': 'Quality Director', 'duration': '1 hour'},
                {'action': 'Conduct On-Site Inspection', 'details': 'Perform physical inspection of processes and documentation', 'responsible_party': 'Audit Team', 'duration': '1-2 days'},
                {'action': 'Document Findings', 'details': 'Record non-conformances and improvement opportunities', 'responsible_party': 'Lead Auditor', 'duration': '4 hours'},
                {'action': 'Present Audit Report', 'details': 'Brief stakeholders on findings and required corrective actions', 'responsible_party': 'Quality Manager', 'duration': '2 hours'},
                {'action': 'Track Corrective Actions', 'details': 'Monitor implementation of required fixes', 'responsible_party': 'Quality Analyst', 'duration': 'Ongoing'}
            ],
            'testing_protocol': [
                {'action': 'Review Testing Requirements', 'details': 'Identify applicable testing standards for {entities}', 'responsible_party': 'Quality Engineer', 'duration': '2 hours'},
                {'action': 'Prepare Test Environment', 'details': 'Set up testing equipment and calibrate instruments', 'responsible_party': 'Lab Technician', 'duration': '4 hours'},
                {'action': 'Execute Test Plan', 'details': 'Run required tests per specification', 'responsible_party': 'Quality Technician', 'duration': '1-3 days'},
                {'action': 'Analyze Results', 'details': 'Compare test results against acceptance criteria', 'responsible_party': 'Quality Engineer', 'duration': '4 hours'},
                {'action': 'Issue Test Report', 'details': 'Document results and disposition recommendations', 'responsible_party': 'Quality Manager', 'duration': '2 hours'}
            ],
            'compliance_action': [
                {'action': 'Identify Compliance Gap', 'details': 'Review regulatory requirements versus current state for {entities}', 'responsible_party': 'Compliance Officer', 'duration': '4 hours'},
                {'action': 'Develop Remediation Plan', 'details': 'Create action plan to address compliance deficiencies', 'responsible_party': 'Compliance/Operations', 'duration': '1 day'},
                {'action': 'Implement Corrective Measures', 'details': 'Execute required changes to processes or documentation', 'responsible_party': 'Operations Team', 'duration': '1-5 days'},
                {'action': 'Validate Compliance', 'details': 'Verify corrective actions meet regulatory requirements', 'responsible_party': 'Compliance Officer', 'duration': '4 hours'},
                {'action': 'Update Documentation', 'details': 'Revise SOPs and compliance records', 'responsible_party': 'Quality/Compliance', 'duration': '2 hours'}
            ],
            'certification_review': [
                {'action': 'Audit Current Certifications', 'details': 'Review status and expiration dates of all relevant certifications', 'responsible_party': 'Compliance Manager', 'duration': '2 hours'},
                {'action': 'Identify Gaps', 'details': 'Determine which certifications need renewal or update', 'responsible_party': 'Compliance Analyst', 'duration': '2 hours'},
                {'action': 'Prepare Documentation', 'details': 'Gather required evidence and records for certification body', 'responsible_party': 'Quality Team', 'duration': '1-2 days'},
                {'action': 'Schedule Certification Audit', 'details': 'Coordinate with certification body for assessment', 'responsible_party': 'Compliance Manager', 'duration': '4 hours'},
                {'action': 'Address Audit Findings', 'details': 'Resolve any non-conformances identified during certification audit', 'responsible_party': 'Operations/Quality', 'duration': '1-5 days'}
            ],
            # Delivery categories
            'route_optimization': [
                {'action': 'Analyze Current Routes', 'details': 'Map existing delivery routes and identify bottlenecks', 'responsible_party': 'Logistics Analyst', 'duration': '4 hours'},
                {'action': 'Evaluate Alternatives', 'details': 'Research alternative routes considering traffic, distance, and costs', 'responsible_party': 'Logistics Planner', 'duration': '2 hours'},
                {'action': 'Test New Routes', 'details': 'Run pilot deliveries on optimized routes', 'responsible_party': 'Delivery Team', 'duration': '1 day'},
                {'action': 'Update Routing Systems', 'details': 'Program new routes into TMS/GPS systems', 'responsible_party': 'IT/Logistics', 'duration': '2 hours'},
                {'action': 'Monitor Performance', 'details': 'Track delivery times and costs on new routes', 'responsible_party': 'Logistics Manager', 'duration': 'Ongoing'}
            ],
            'carrier_switch': [
                {'action': 'Evaluate Alternative Carriers', 'details': 'Research available carriers with capacity for {entities}', 'responsible_party': 'Logistics Manager', 'duration': '4 hours'},
                {'action': 'Request Rate Quotes', 'details': 'Obtain pricing from shortlisted carriers', 'responsible_party': 'Logistics Analyst', 'duration': '2 hours'},
                {'action': 'Negotiate Terms', 'details': 'Finalize service agreements and SLAs', 'responsible_party': 'Logistics Manager', 'duration': '4 hours'},
                {'action': 'Onboard New Carrier', 'details': 'Set up carrier in TMS and provide routing instructions', 'responsible_party': 'Logistics/IT', 'duration': '2 hours'},
                {'action': 'Transition Shipments', 'details': 'Redirect pending shipments to new carrier', 'responsible_party': 'Logistics Coordinator', 'duration': '1 day'}
            ],
            'delivery_rescheduling': [
                {'action': 'Assess Delivery Impact', 'details': 'Identify all affected deliveries and customers', 'responsible_party': 'Customer Service', 'duration': '2 hours'},
                {'action': 'Develop Revised Schedule', 'details': 'Create new delivery timeline based on current constraints', 'responsible_party': 'Logistics Planner', 'duration': '2 hours'},
                {'action': 'Notify Customers', 'details': 'Proactively communicate delays and new ETAs to affected customers', 'responsible_party': 'Customer Service', 'duration': '4 hours'},
                {'action': 'Offer Compensation', 'details': 'Provide appropriate remediation (discounts, expedited shipping credits)', 'responsible_party': 'Customer Service Manager', 'duration': '2 hours'},
                {'action': 'Confirm New Deliveries', 'details': 'Verify rescheduled deliveries are on track', 'responsible_party': 'Logistics Coordinator', 'duration': 'Ongoing'}
            ],
            'packaging_redesign': [
                {'action': 'Analyze Packaging Failures', 'details': 'Review damage reports to identify packaging weaknesses', 'responsible_party': 'Packaging Engineer', 'duration': '4 hours'},
                {'action': 'Research Solutions', 'details': 'Evaluate improved packaging materials and designs', 'responsible_party': 'Packaging/R&D', 'duration': '1 day'},
                {'action': 'Test New Packaging', 'details': 'Conduct drop tests and transit simulations', 'responsible_party': 'Quality/Packaging', 'duration': '1-2 days'},
                {'action': 'Source Materials', 'details': 'Order improved packaging materials from suppliers', 'responsible_party': 'Procurement', 'duration': '3-5 days'},
                {'action': 'Implement Changes', 'details': 'Roll out new packaging across affected product lines', 'responsible_party': 'Operations', 'duration': '1 week'}
            ],
            # Production categories
            'equipment_repair': [
                {'action': 'Diagnose Equipment Issue', 'details': 'Identify root cause of equipment failure for {entities}', 'responsible_party': 'Maintenance Technician', 'duration': '2-4 hours'},
                {'action': 'Source Replacement Parts', 'details': 'Locate required parts in inventory or order emergency delivery', 'responsible_party': 'Maintenance/Procurement', 'duration': '2 hours - 2 days'},
                {'action': 'Perform Repair', 'details': 'Execute repair procedure following manufacturer guidelines', 'responsible_party': 'Maintenance Team', 'duration': '2-8 hours'},
                {'action': 'Test Equipment', 'details': 'Run validation tests to confirm proper operation', 'responsible_party': 'Maintenance/Quality', 'duration': '1-2 hours'},
                {'action': 'Return to Production', 'details': 'Clear equipment for production use and update maintenance logs', 'responsible_party': 'Production Manager', 'duration': '30 minutes'}
            ],
            'maintenance_schedule': [
                {'action': 'Review Maintenance History', 'details': 'Analyze past maintenance records for {entities}', 'responsible_party': 'Maintenance Planner', 'duration': '2 hours'},
                {'action': 'Update Maintenance Plan', 'details': 'Revise PM schedules based on failure patterns', 'responsible_party': 'Maintenance Manager', 'duration': '4 hours'},
                {'action': 'Allocate Resources', 'details': 'Schedule technicians and parts for upcoming maintenance', 'responsible_party': 'Maintenance Planner', 'duration': '2 hours'},
                {'action': 'Execute Maintenance', 'details': 'Perform scheduled preventive maintenance activities', 'responsible_party': 'Maintenance Team', 'duration': '4-8 hours'},
                {'action': 'Document Results', 'details': 'Update CMMS with completed work and findings', 'responsible_party': 'Maintenance Technician', 'duration': '30 minutes'}
            ],
            'capacity_adjustment': [
                {'action': 'Assess Current Capacity', 'details': 'Review production output and bottlenecks', 'responsible_party': 'Production Manager', 'duration': '2 hours'},
                {'action': 'Identify Adjustment Options', 'details': 'Evaluate overtime, shift changes, or outsourcing', 'responsible_party': 'Operations Director', 'duration': '2 hours'},
                {'action': 'Implement Capacity Changes', 'details': 'Execute selected capacity adjustment strategy', 'responsible_party': 'Production/HR', 'duration': '1-3 days'},
                {'action': 'Monitor Output', 'details': 'Track production rates against targets', 'responsible_party': 'Production Supervisor', 'duration': 'Ongoing'},
                {'action': 'Adjust as Needed', 'details': 'Fine-tune capacity based on actual performance', 'responsible_party': 'Production Manager', 'duration': 'Ongoing'}
            ],
            'workflow_optimization': [
                {'action': 'Map Current Workflow', 'details': 'Document existing process steps and cycle times', 'responsible_party': 'Process Engineer', 'duration': '4 hours'},
                {'action': 'Identify Waste', 'details': 'Apply lean principles to find non-value-added activities', 'responsible_party': 'Lean Specialist', 'duration': '4 hours'},
                {'action': 'Design Improvements', 'details': 'Create optimized workflow with reduced waste', 'responsible_party': 'Process Engineer', 'duration': '1 day'},
                {'action': 'Pilot Changes', 'details': 'Test new workflow on limited scope', 'responsible_party': 'Production Team', 'duration': '1-2 days'},
                {'action': 'Roll Out', 'details': 'Implement optimized workflow across production', 'responsible_party': 'Operations Manager', 'duration': '1 week'}
            ],
            # Brand categories
            'crisis_communication': [
                {'action': 'Activate Crisis Team', 'details': 'Assemble crisis response team and establish command center', 'responsible_party': 'Communications Director', 'duration': '1 hour'},
                {'action': 'Assess Situation', 'details': 'Gather facts and determine scope of brand impact', 'responsible_party': 'Crisis Team', 'duration': '2 hours'},
                {'action': 'Develop Key Messages', 'details': 'Craft approved messaging for all stakeholder groups', 'responsible_party': 'PR/Legal', 'duration': '2-4 hours'},
                {'action': 'Brief Spokespersons', 'details': 'Prepare designated spokespersons with talking points', 'responsible_party': 'Communications Director', 'duration': '1 hour'},
                {'action': 'Execute Communication Plan', 'details': 'Distribute messaging through appropriate channels', 'responsible_party': 'Communications Team', 'duration': '2 hours'},
                {'action': 'Monitor Response', 'details': 'Track media and social media reaction', 'responsible_party': 'PR Team', 'duration': 'Ongoing'}
            ],
            'pr_campaign': [
                {'action': 'Define Campaign Objectives', 'details': 'Establish clear goals for brand recovery', 'responsible_party': 'Marketing Director', 'duration': '2 hours'},
                {'action': 'Develop Campaign Strategy', 'details': 'Create messaging and channel strategy', 'responsible_party': 'PR Agency/Team', 'duration': '1-2 days'},
                {'action': 'Create Content', 'details': 'Produce press releases, social content, and collateral', 'responsible_party': 'Content Team', 'duration': '2-5 days'},
                {'action': 'Launch Campaign', 'details': 'Execute across planned channels and media outlets', 'responsible_party': 'PR Team', 'duration': '1 day'},
                {'action': 'Measure Results', 'details': 'Track reach, sentiment, and brand perception metrics', 'responsible_party': 'Marketing Analytics', 'duration': 'Ongoing'}
            ],
            'stakeholder_outreach': [
                {'action': 'Identify Key Stakeholders', 'details': 'Prioritize stakeholders requiring direct communication', 'responsible_party': 'Executive Team', 'duration': '1 hour'},
                {'action': 'Prepare Briefing Materials', 'details': 'Create stakeholder-specific messaging and FAQs', 'responsible_party': 'Communications/IR', 'duration': '4 hours'},
                {'action': 'Schedule Meetings', 'details': 'Arrange calls or meetings with priority stakeholders', 'responsible_party': 'Executive Assistant', 'duration': '2 hours'},
                {'action': 'Conduct Briefings', 'details': 'Deliver personalized updates to each stakeholder group', 'responsible_party': 'Executives', 'duration': '1-2 days'},
                {'action': 'Follow Up', 'details': 'Provide additional information as requested', 'responsible_party': 'Investor Relations', 'duration': 'Ongoing'}
            ],
            'reputation_recovery': [
                {'action': 'Assess Reputation Damage', 'details': 'Conduct brand perception survey and media analysis', 'responsible_party': 'Marketing Research', 'duration': '1 week'},
                {'action': 'Develop Recovery Strategy', 'details': 'Create long-term plan to rebuild brand trust', 'responsible_party': 'Marketing/PR', 'duration': '1 week'},
                {'action': 'Implement Quick Wins', 'details': 'Execute immediate actions to demonstrate change', 'responsible_party': 'Operations/Marketing', 'duration': '1-2 weeks'},
                {'action': 'Launch Reputation Campaign', 'details': 'Begin sustained positive messaging and actions', 'responsible_party': 'Marketing Team', 'duration': 'Ongoing'},
                {'action': 'Track Progress', 'details': 'Monitor brand metrics and adjust strategy', 'responsible_party': 'Marketing Analytics', 'duration': 'Quarterly'}
            ],
            # Historical resolution
            'historical_resolution': [
                {'action': 'Review Historical Case', 'details': 'Analyze previous similar risk and resolution approach', 'responsible_party': 'Risk Manager', 'duration': '2 hours'},
                {'action': 'Adapt Solution', 'details': 'Modify historical approach for current context', 'responsible_party': 'Operations Team', 'duration': '2 hours'},
                {'action': 'Implement Resolution', 'details': 'Execute adapted solution steps', 'responsible_party': 'Assigned Team', 'duration': '1-3 days'},
                {'action': 'Document Lessons', 'details': 'Record outcomes to enhance future responses', 'responsible_party': 'Risk Analyst', 'duration': '2 hours'}
            ],
            # External recommendation (Grok)
            'external_recommendation': [
                {'action': 'Review Industry Practice', 'details': 'Study recommended approach from external source', 'responsible_party': 'Subject Matter Expert', 'duration': '2 hours'},
                {'action': 'Assess Applicability', 'details': 'Evaluate how practice applies to your situation', 'responsible_party': 'Operations Manager', 'duration': '2 hours'},
                {'action': 'Develop Implementation Plan', 'details': 'Create action plan to adopt best practice', 'responsible_party': 'Project Manager', 'duration': '4 hours'},
                {'action': 'Execute Plan', 'details': 'Implement recommended practices', 'responsible_party': 'Cross-functional Team', 'duration': '1-2 weeks'},
                {'action': 'Measure Effectiveness', 'details': 'Track results against expected outcomes', 'responsible_party': 'Operations Analyst', 'duration': 'Ongoing'}
            ]
        }

        # Return matching template or generic steps
        if category in templates:
            return templates[category]

        # Default generic steps
        return [
            {'action': 'Assess Situation', 'details': 'Review current status and impact for {entities}', 'responsible_party': 'Team Lead', 'duration': '2 hours'},
            {'action': 'Develop Action Plan', 'details': 'Create specific steps to address the issue', 'responsible_party': 'Manager', 'duration': '4 hours'},
            {'action': 'Execute Solution', 'details': 'Implement planned actions', 'responsible_party': 'Operations Team', 'duration': '1-3 days'},
            {'action': 'Verify Results', 'details': 'Confirm solution effectiveness', 'responsible_party': 'Quality/Manager', 'duration': '2 hours'},
            {'action': 'Document Outcomes', 'details': 'Record resolution for future reference', 'responsible_party': 'Analyst', 'duration': '1 hour'}
        ]

    def _contextualize_step(self, details: str, risk_context: Dict) -> str:
        """Replace placeholders in step details with actual context."""
        entities = risk_context.get('key_entities', [])
        entity_str = ', '.join([e['value'] for e in entities]) if entities else 'affected items'

        return details.replace('{entities}', entity_str)

    def _adjust_duration_for_severity(self, base_duration: str, severity: str) -> str:
        """Adjust estimated duration based on severity (faster for critical)."""
        if severity == 'CRITICAL':
            # Expedite timelines for critical severity
            if 'day' in base_duration:
                return base_duration.replace('days', 'day').replace('1 day', '4-8 hours')
            elif 'week' in base_duration:
                return base_duration.replace('weeks', 'week').replace('1 week', '2-3 days')
        elif severity == 'HIGH':
            if 'week' in base_duration:
                return base_duration.replace('weeks', 'week')
        return base_duration

    def _generate_expected_outcome(self, solution: Dict, risk_context: Dict) -> str:
        """Generate expected outcome statement for the solution."""
        category = solution.get('solution_category', 'general')
        title = solution.get('title', 'Solution')

        outcomes = {
            'supplier_engagement': 'Improved supplier relationship and confirmed delivery commitments',
            'inventory_reallocation': 'Optimized inventory distribution and reduced stockout risk',
            'alternative_sourcing': 'Diversified supply base and reduced single-source dependency',
            'contract_renegotiation': 'Updated contract terms with improved protections',
            'quality_audit': 'Identified quality gaps with actionable improvement plan',
            'testing_protocol': 'Validated product quality with documented test results',
            'compliance_action': 'Achieved regulatory compliance with documented evidence',
            'certification_review': 'Current certifications maintained or renewed',
            'route_optimization': 'Reduced delivery times and transportation costs',
            'carrier_switch': 'Improved delivery reliability with new carrier partnership',
            'delivery_rescheduling': 'Customers informed with revised delivery expectations',
            'packaging_redesign': 'Reduced product damage during transit',
            'equipment_repair': 'Equipment restored to full operational capacity',
            'maintenance_schedule': 'Optimized preventive maintenance reducing unplanned downtime',
            'capacity_adjustment': 'Production capacity aligned with demand requirements',
            'workflow_optimization': 'Improved efficiency and reduced production waste',
            'crisis_communication': 'Controlled narrative with stakeholders informed',
            'pr_campaign': 'Improved brand perception and positive media coverage',
            'stakeholder_outreach': 'Key stakeholders engaged and supportive',
            'reputation_recovery': 'Restored brand trust with measurable improvement'
        }

        return outcomes.get(category, f'Successful implementation of {title} with risk mitigation achieved')

    def _estimate_timeline(self, solution: Dict, severity: str, complexity_score: int) -> str:
        """Estimate overall timeline for solution implementation."""
        urgency = solution.get('urgency', 'STANDARD')

        # Base timelines by urgency
        timelines = {
            'IMMEDIATE': '24-48 hours',
            'URGENT': '3-5 days',
            'STANDARD': '1-2 weeks',
            'PLANNED': '2-4 weeks'
        }

        base = timelines.get(urgency, '1-2 weeks')

        # Adjust for complexity
        if complexity_score >= 8:
            if urgency in ['STANDARD', 'PLANNED']:
                return base.replace('weeks', 'weeks (extended due to complexity)')

        return base

    def _query_supabase_solutions(
        self,
        risk_type: str,
        severity: str,
        description: str,
        structured_data: Dict = None
    ) -> List[Dict]:
        """
        Query Supabase for risk-type-specific solutions from private data.
        Returns solutions with doc_id references for attachments.

        Queries different tables based on risk type:
        - SUPPLY_CHAIN: suppliers, inventory_items, contracts, logistics_providers
        - QUALITY: quality_standards, certifications, testing_equipment, audit_history
        - DELIVERY: logistics_providers, carriers, shipping_routes, packaging_options
        - PRODUCTION: equipment, spare_parts, maintenance_schedules, production_lines
        - BRAND: pr_agencies, communication_templates, stakeholder_contacts, media_contacts
        """
        solutions = []

        if not self.supabase:
            return solutions

        risk_config = self.RISK_TYPE_SOLUTIONS.get(risk_type, self.RISK_TYPE_SOLUTIONS['SUPPLY_CHAIN'])

        try:
            # ===== RISK-TYPE-SPECIFIC QUERIES =====

            if risk_type == 'SUPPLY_CHAIN':
                solutions.extend(self._query_supply_chain_solutions(severity))

            elif risk_type == 'QUALITY':
                solutions.extend(self._query_quality_solutions(severity))

            elif risk_type == 'DELIVERY':
                solutions.extend(self._query_delivery_solutions(severity))

            elif risk_type == 'PRODUCTION':
                solutions.extend(self._query_production_solutions(severity))

            elif risk_type == 'BRAND':
                solutions.extend(self._query_brand_solutions(severity))

            # ===== HISTORICAL RESOLUTIONS (All Risk Types) =====
            historical_result = self.supabase.table('risks').select(
                'id, description, risk_type, severity, structured_data'
            ).eq('risk_type', risk_type).limit(5).execute()

            if historical_result.data:
                for record in historical_result.data:
                    if record.get('structured_data', {}).get('resolution'):
                        solutions.append({
                            'title': f"Historical Resolution: {record['risk_type']}",
                            'description': f"Previously resolved similar {record['risk_type']} risk: "
                                          f"{record['description'][:100]}...",
                            'source': 'supabase',
                            'source_type': 'Private Data',
                            'reference': record['id'],
                            'reference_type': 'document',
                            'urgency': self._severity_to_urgency(record.get('severity', 'MEDIUM')),
                            'confidence': 0.85,
                            'success_probability': 0.85,
                            'solution_category': 'historical_resolution',
                            'historical_context': {
                                'original_severity': record.get('severity'),
                                'risk_type': record.get('risk_type')
                            }
                        })

        except Exception as e:
            print(f"Supabase solution query error: {e}")

        return solutions

    def _severity_to_urgency(self, severity: str) -> str:
        """Convert severity to urgency level."""
        return self.SEVERITY_TO_URGENCY.get(severity, 'STANDARD')

    def _lead_time_to_urgency(self, lead_days: int) -> str:
        """Convert lead time days to urgency level."""
        if lead_days <= 1:
            return 'IMMEDIATE'
        elif lead_days <= 3:
            return 'URGENT'
        elif lead_days <= 7:
            return 'STANDARD'
        return 'PLANNED'

    def _query_supply_chain_solutions(self, severity: str) -> List[Dict]:
        """Query SUPPLY_CHAIN specific solutions: raw_materials, components, logistics."""
        solutions = []

        # Suppliers
        try:
            result = self.supabase.table('suppliers').select(
                'id, name, category, lead_time_days, reliability_score, '
                'is_backup, pricing_tier, certifications, contact_info'
            ).execute()

            if result.data:
                for supplier in result.data:
                    lead_days = supplier.get('lead_time_days', 30)
                    solutions.append({
                        'title': f"Engage Supplier: {supplier['name']}",
                        'description': f"Contact {supplier['name']} ({supplier.get('category', 'general')}) - "
                                      f"Lead time: {lead_days} days, Reliability: {supplier.get('reliability_score', 0):.0%}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': supplier['id'],
                        'reference_type': 'document',
                        'urgency': self._lead_time_to_urgency(lead_days),
                        'confidence': supplier.get('reliability_score', 0.7),
                        'success_probability': supplier.get('reliability_score', 0.7),
                        'solution_category': 'supplier_engagement',
                        'supplier_details': {
                            'name': supplier['name'],
                            'category': supplier.get('category'),
                            'pricing_tier': supplier.get('pricing_tier'),
                            'certifications': supplier.get('certifications', []),
                            'is_backup': supplier.get('is_backup', False)
                        }
                    })
        except Exception as e:
            print(f"Suppliers query error: {e}")

        # Inventory Items
        try:
            result = self.supabase.table('inventory_items').select(
                'id, name, sku, category, current_stock, reorder_point, '
                'unit_cost, lead_time_days, is_critical'
            ).gt('current_stock', 0).execute()

            if result.data:
                for item in result.data:
                    confidence = 0.9 if item.get('is_critical') else 0.75
                    solutions.append({
                        'title': f"Deploy from Inventory: {item['name']}",
                        'description': f"Available: {item['current_stock']} units (SKU: {item.get('sku', 'N/A')}) - "
                                      f"Unit cost: ${item.get('unit_cost', 0):.2f}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': item['id'],
                        'reference_type': 'document',
                        'urgency': self._lead_time_to_urgency(item.get('lead_time_days', 7)),
                        'confidence': confidence,
                        'success_probability': confidence,
                        'solution_category': 'inventory_reallocation',
                        'inventory_details': {
                            'sku': item.get('sku'),
                            'current_stock': item.get('current_stock'),
                            'unit_cost': item.get('unit_cost'),
                            'is_critical': item.get('is_critical', False)
                        }
                    })
        except Exception as e:
            print(f"Inventory query error: {e}")

        # Logistics Providers
        try:
            result = self.supabase.table('logistics_providers').select('*').execute()
            if result.data:
                for provider in result.data:
                    solutions.append({
                        'title': f"Logistics Partner: {provider.get('name', 'Provider')}",
                        'description': f"Engage {provider.get('name')} for logistics support - "
                                      f"Coverage: {provider.get('coverage_area', 'Regional')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': provider['id'],
                        'reference_type': 'document',
                        'urgency': self._lead_time_to_urgency(provider.get('response_time_days', 3)),
                        'confidence': provider.get('reliability_score', 0.8),
                        'success_probability': provider.get('reliability_score', 0.8),
                        'solution_category': 'logistics',
                        'logistics_details': provider
                    })
        except Exception:
            pass  # Table may not exist

        return solutions

    def _query_quality_solutions(self, severity: str) -> List[Dict]:
        """Query QUALITY specific solutions: testing_equipment, quality_tools, certifications."""
        solutions = []

        # Quality Standards
        try:
            result = self.supabase.table('quality_standards').select('*').execute()
            if result.data:
                for standard in result.data:
                    solutions.append({
                        'title': f"Apply Standard: {standard.get('name', 'Quality Standard')}",
                        'description': f"Implement {standard.get('name')} - {standard.get('description', '')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': standard['id'],
                        'reference_type': 'document',
                        'urgency': 'STANDARD' if severity in ['LOW', 'MEDIUM'] else 'URGENT',
                        'confidence': 0.85,
                        'success_probability': 0.85,
                        'solution_category': 'quality_audit',
                        'standard_details': standard
                    })
        except Exception:
            pass

        # Certifications
        try:
            result = self.supabase.table('certifications').select('*').execute()
            if result.data:
                for cert in result.data:
                    solutions.append({
                        'title': f"Certification Review: {cert.get('name', 'Certification')}",
                        'description': f"Review/renew {cert.get('name')} certification - "
                                      f"Expires: {cert.get('expiry_date', 'N/A')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': cert['id'],
                        'reference_type': 'document',
                        'urgency': 'URGENT' if cert.get('is_expiring') else 'PLANNED',
                        'confidence': 0.9,
                        'success_probability': 0.9,
                        'solution_category': 'certification_review',
                        'certification_details': cert
                    })
        except Exception:
            pass

        # Testing Equipment
        try:
            result = self.supabase.table('testing_equipment').select('*').execute()
            if result.data:
                for equipment in result.data:
                    solutions.append({
                        'title': f"Deploy Testing: {equipment.get('name', 'Equipment')}",
                        'description': f"Use {equipment.get('name')} for quality testing - "
                                      f"Type: {equipment.get('test_type', 'General')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': equipment['id'],
                        'reference_type': 'document',
                        'urgency': 'URGENT' if severity == 'CRITICAL' else 'STANDARD',
                        'confidence': equipment.get('accuracy_rating', 0.85),
                        'success_probability': equipment.get('accuracy_rating', 0.85),
                        'solution_category': 'testing_protocol',
                        'equipment_details': equipment
                    })
        except Exception:
            pass

        return solutions

    def _query_delivery_solutions(self, severity: str) -> List[Dict]:
        """Query DELIVERY specific solutions: logistics, packaging, transportation."""
        solutions = []

        # Carriers
        try:
            result = self.supabase.table('carriers').select('*').execute()
            if result.data:
                for carrier in result.data:
                    solutions.append({
                        'title': f"Switch Carrier: {carrier.get('name', 'Carrier')}",
                        'description': f"Use {carrier.get('name')} - "
                                      f"Transit time: {carrier.get('transit_days', 'N/A')} days, "
                                      f"Coverage: {carrier.get('coverage', 'Regional')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': carrier['id'],
                        'reference_type': 'document',
                        'urgency': self._lead_time_to_urgency(carrier.get('transit_days', 5)),
                        'confidence': carrier.get('reliability_score', 0.8),
                        'success_probability': carrier.get('on_time_rate', 0.85),
                        'solution_category': 'carrier_switch',
                        'carrier_details': carrier
                    })
        except Exception:
            pass

        # Shipping Routes
        try:
            result = self.supabase.table('shipping_routes').select('*').execute()
            if result.data:
                for route in result.data:
                    solutions.append({
                        'title': f"Alternative Route: {route.get('name', 'Route')}",
                        'description': f"Use {route.get('name')} route - "
                                      f"From: {route.get('origin')} To: {route.get('destination')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': route['id'],
                        'reference_type': 'document',
                        'urgency': 'IMMEDIATE' if route.get('is_express') else 'STANDARD',
                        'confidence': 0.8,
                        'success_probability': route.get('success_rate', 0.85),
                        'solution_category': 'route_optimization',
                        'route_details': route
                    })
        except Exception:
            pass

        # Packaging Options
        try:
            result = self.supabase.table('packaging_options').select('*').execute()
            if result.data:
                for pkg in result.data:
                    solutions.append({
                        'title': f"Packaging Option: {pkg.get('name', 'Package')}",
                        'description': f"Use {pkg.get('name')} packaging - "
                                      f"Protection level: {pkg.get('protection_level', 'Standard')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': pkg['id'],
                        'reference_type': 'document',
                        'urgency': 'STANDARD',
                        'confidence': 0.85,
                        'success_probability': 0.9,
                        'solution_category': 'packaging_redesign',
                        'packaging_details': pkg
                    })
        except Exception:
            pass

        return solutions

    def _query_production_solutions(self, severity: str) -> List[Dict]:
        """Query PRODUCTION specific solutions: equipment, machinery, spare_parts, maintenance."""
        solutions = []

        # Equipment
        try:
            result = self.supabase.table('equipment').select('*').execute()
            if result.data:
                for equip in result.data:
                    solutions.append({
                        'title': f"Equipment Action: {equip.get('name', 'Equipment')}",
                        'description': f"Service/repair {equip.get('name')} - "
                                      f"Status: {equip.get('status', 'Unknown')}, "
                                      f"Last maintenance: {equip.get('last_maintenance', 'N/A')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': equip['id'],
                        'reference_type': 'document',
                        'urgency': 'IMMEDIATE' if equip.get('status') == 'critical' else 'STANDARD',
                        'confidence': 0.85,
                        'success_probability': 0.85,
                        'solution_category': 'equipment_repair',
                        'equipment_details': equip
                    })
        except Exception:
            pass

        # Spare Parts
        try:
            result = self.supabase.table('spare_parts').select('*').gt('quantity', 0).execute()
            if result.data:
                for part in result.data:
                    solutions.append({
                        'title': f"Deploy Spare Part: {part.get('name', 'Part')}",
                        'description': f"Available: {part.get('quantity', 0)} units of {part.get('name')} - "
                                      f"Compatible with: {part.get('compatible_equipment', 'N/A')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': part['id'],
                        'reference_type': 'document',
                        'urgency': 'IMMEDIATE' if part.get('is_critical') else 'STANDARD',
                        'confidence': 0.9,
                        'success_probability': 0.9,
                        'solution_category': 'spare_parts',
                        'part_details': part
                    })
        except Exception:
            pass

        # Maintenance Schedules
        try:
            result = self.supabase.table('maintenance_schedules').select('*').execute()
            if result.data:
                for schedule in result.data:
                    solutions.append({
                        'title': f"Maintenance Schedule: {schedule.get('name', 'Schedule')}",
                        'description': f"Execute {schedule.get('name')} maintenance protocol - "
                                      f"Frequency: {schedule.get('frequency', 'N/A')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': schedule['id'],
                        'reference_type': 'document',
                        'urgency': 'URGENT' if schedule.get('is_overdue') else 'PLANNED',
                        'confidence': 0.85,
                        'success_probability': 0.9,
                        'solution_category': 'maintenance_schedule',
                        'schedule_details': schedule
                    })
        except Exception:
            pass

        return solutions

    def _query_brand_solutions(self, severity: str) -> List[Dict]:
        """Query BRAND specific solutions: marketing, pr_services, communications."""
        solutions = []

        # PR Agencies
        try:
            result = self.supabase.table('pr_agencies').select('*').execute()
            if result.data:
                for agency in result.data:
                    solutions.append({
                        'title': f"Engage PR Agency: {agency.get('name', 'Agency')}",
                        'description': f"Contact {agency.get('name')} for crisis management - "
                                      f"Specialty: {agency.get('specialty', 'General PR')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': agency['id'],
                        'reference_type': 'document',
                        'urgency': 'IMMEDIATE' if severity == 'CRITICAL' else 'URGENT',
                        'confidence': agency.get('success_rate', 0.8),
                        'success_probability': agency.get('success_rate', 0.8),
                        'solution_category': 'pr_campaign',
                        'agency_details': agency
                    })
        except Exception:
            pass

        # Communication Templates
        try:
            result = self.supabase.table('communication_templates').select('*').execute()
            if result.data:
                for template in result.data:
                    solutions.append({
                        'title': f"Use Template: {template.get('name', 'Template')}",
                        'description': f"Deploy {template.get('name')} communication template - "
                                      f"Type: {template.get('type', 'General')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': template['id'],
                        'reference_type': 'document',
                        'urgency': 'IMMEDIATE' if template.get('type') == 'crisis' else 'STANDARD',
                        'confidence': 0.85,
                        'success_probability': 0.85,
                        'solution_category': 'crisis_communication',
                        'template_details': template
                    })
        except Exception:
            pass

        # Stakeholder Contacts
        try:
            result = self.supabase.table('stakeholder_contacts').select('*').execute()
            if result.data:
                for contact in result.data:
                    solutions.append({
                        'title': f"Contact Stakeholder: {contact.get('name', 'Stakeholder')}",
                        'description': f"Reach out to {contact.get('name')} ({contact.get('role', 'Stakeholder')}) - "
                                      f"Priority: {contact.get('priority', 'Standard')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': contact['id'],
                        'reference_type': 'document',
                        'urgency': 'IMMEDIATE' if contact.get('priority') == 'high' else 'URGENT',
                        'confidence': 0.9,
                        'success_probability': 0.85,
                        'solution_category': 'stakeholder_outreach',
                        'contact_details': contact
                    })
        except Exception:
            pass

        # Media Contacts
        try:
            result = self.supabase.table('media_contacts').select('*').execute()
            if result.data:
                for media in result.data:
                    solutions.append({
                        'title': f"Media Contact: {media.get('name', 'Media')}",
                        'description': f"Engage {media.get('name')} ({media.get('outlet', 'Media')}) - "
                                      f"Beat: {media.get('beat', 'General')}",
                        'source': 'supabase',
                        'source_type': 'Private Data',
                        'reference': media['id'],
                        'reference_type': 'document',
                        'urgency': 'URGENT',
                        'confidence': 0.75,
                        'success_probability': 0.7,
                        'solution_category': 'reputation_recovery',
                        'media_details': media
                    })
        except Exception:
            pass

        return solutions

    def _query_grok_solutions(
        self,
        risk_type: str,
        severity: str,
        description: str,
        complexity_score: int
    ) -> List[Dict]:
        """
        Query Grok for risk-type-specific external intelligence solutions.
        Returns solutions with URL references to external resources.

        Each solution includes:
        - Summary and description
        - Step-by-step instructions
        - Expected outcome
        - Source attribution

        ALWAYS returns solutions - uses risk-type-specific prompts for better results.
        Both Supabase and Grok run simultaneously, never as fallbacks.
        """
        solutions = []

        # Get risk-specific categories for the prompt
        risk_config = self.RISK_TYPE_SOLUTIONS.get(risk_type, self.RISK_TYPE_SOLUTIONS['SUPPLY_CHAIN'])
        categories = risk_config['categories']
        solution_types = risk_config['solution_types']

        # Build risk-type-specific prompt
        prompt = self._build_grok_prompt(risk_type, severity, description, complexity_score, categories, solution_types)

        # Query Grok API
        if self.grok_engine and self.grok_engine.api_key:
            try:
                response = requests.post(
                    self.grok_engine.grok_url,
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "model": "grok-2",
                        "temperature": 0.4
                    },
                    headers={"Authorization": f"Bearer {self.grok_engine.api_key}"},
                    timeout=25
                )

                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']

                    # Parse JSON from response
                    match = re.search(r'\[.*\]', content, re.DOTALL)
                    if match:
                        grok_solutions = json.loads(match.group())

                        for sol in grok_solutions:
                            # Parse steps from Grok response (may be provided or empty)
                            grok_steps = sol.get('steps', [])

                            solutions.append({
                                'title': sol.get('title', 'External Recommendation'),
                                'summary': sol.get('summary', ''),
                                'description': sol.get('description', ''),
                                'steps': grok_steps,  # Steps from Grok (will be enriched later if empty)
                                'expected_outcome': sol.get('expected_outcome', ''),
                                'source': 'grok',
                                'source_type': 'External Intelligence',
                                'reference': sol.get('url', 'https://x.ai/grok'),
                                'reference_type': 'url',
                                'urgency': sol.get('urgency', 'STANDARD').upper(),
                                'confidence': 0.75,
                                'success_probability': sol.get('success_probability', 0.7),
                                'solution_category': sol.get('category', 'external_recommendation')
                            })

            except Exception as e:
                print(f"Grok API error: {e}")

        # ALWAYS add risk-type-specific external solutions (Grok + industry best practices)
        # These complement any API responses - not a fallback
        solutions.extend(self._get_risk_type_external_solutions(risk_type, severity))

        return solutions

    def _build_grok_prompt(
        self,
        risk_type: str,
        severity: str,
        description: str,
        complexity_score: int,
        categories: List[str],
        solution_types: List[str]
    ) -> str:
        """Build risk-type-specific Grok prompt with step-by-step instructions."""

        category_str = ', '.join(categories)
        solution_type_str = ', '.join(solution_types)

        # Determine number of steps based on complexity
        if complexity_score <= 3:
            num_steps = "3"
        elif complexity_score <= 6:
            num_steps = "4-5"
        else:
            num_steps = "5-7"

        return f"""Analyze this {risk_type} business risk and provide specific, actionable solutions with step-by-step instructions.

RISK DETAILS:
- Type: {risk_type}
- Severity: {severity}
- Complexity: {complexity_score}/10
- Description: {description}

SOLUTION CATEGORIES TO FOCUS ON: {category_str}
SOLUTION TYPES NEEDED: {solution_type_str}

For each solution, provide:
1. A clear action title specific to {risk_type}
2. A brief summary (1-2 sentences) explaining what this solution accomplishes
3. Step-by-step instructions ({num_steps} steps based on complexity)
4. Expected outcome when successfully implemented
5. Solution category (one of: {solution_type_str})
6. Urgency level: IMMEDIATE (crisis), URGENT (priority), STANDARD (normal), or PLANNED (strategic)
7. A relevant URL to an authoritative industry source
8. Estimated success probability (0.0-1.0)

Return JSON array:
[
    {{
        "title": "Solution title",
        "summary": "Brief overview of what this solution accomplishes",
        "description": "Detailed context and rationale",
        "steps": [
            {{"step": 1, "action": "First action", "details": "Specific instructions", "responsible_party": "Role/Team", "duration": "Estimated time"}},
            {{"step": 2, "action": "Second action", "details": "Specific instructions", "responsible_party": "Role/Team", "duration": "Estimated time"}}
        ],
        "expected_outcome": "What success looks like",
        "category": "{solution_types[0]}",
        "urgency": "IMMEDIATE|URGENT|STANDARD|PLANNED",
        "url": "https://example.com/resource",
        "success_probability": 0.85
    }}
]

Provide 4-6 solutions across different urgency tiers, focusing on {risk_type} best practices. Tailor each solution to the specific risk described."""

    def _get_risk_type_external_solutions(self, risk_type: str, severity: str) -> List[Dict]:
        """
        Get risk-type-specific external solutions with industry URLs.
        These ALWAYS run alongside Grok API - not as fallback.
        """

        external_solutions = {
            'SUPPLY_CHAIN': [
                {
                    'title': 'Implement Supplier Risk Assessment',
                    'description': 'Conduct comprehensive supplier risk evaluation using industry-standard frameworks.',
                    'category': 'supplier_engagement',
                    'urgency': 'URGENT',
                    'url': 'https://www.mckinsey.com/capabilities/operations/our-insights/supply-chain-resilience',
                    'success_probability': 0.88
                },
                {
                    'title': 'Dual-Sourcing Strategy',
                    'description': 'Establish contracts with 2+ suppliers for critical components to reduce single-source risk.',
                    'category': 'alternative_sourcing',
                    'urgency': 'STANDARD',
                    'url': 'https://www.gartner.com/en/supply-chain/topics/supply-chain-risk-management',
                    'success_probability': 0.90
                },
                {
                    'title': 'Supply Chain Visibility Platform',
                    'description': 'Implement real-time tracking and monitoring across the supply chain.',
                    'category': 'logistics',
                    'urgency': 'PLANNED',
                    'url': 'https://www.supplychainbrain.com/articles/best-practices-supply-chain-visibility',
                    'success_probability': 0.85
                },
                {
                    'title': 'Emergency Procurement Protocol',
                    'description': 'Activate emergency sourcing procedures for critical materials.',
                    'category': 'inventory_reallocation',
                    'urgency': 'IMMEDIATE',
                    'url': 'https://www.cips.org/knowledge/procurement-topics/emergency-procurement',
                    'success_probability': 0.80
                }
            ],
            'QUALITY': [
                {
                    'title': 'ISO 9001 Compliance Audit',
                    'description': 'Conduct comprehensive quality management system audit per ISO 9001:2015 standards.',
                    'category': 'quality_audit',
                    'urgency': 'URGENT',
                    'url': 'https://www.iso.org/iso-9001-quality-management.html',
                    'success_probability': 0.92
                },
                {
                    'title': 'Six Sigma Quality Improvement',
                    'description': 'Apply Six Sigma DMAIC methodology to identify and eliminate defects.',
                    'category': 'testing_protocol',
                    'urgency': 'STANDARD',
                    'url': 'https://asq.org/quality-resources/six-sigma',
                    'success_probability': 0.88
                },
                {
                    'title': 'Root Cause Analysis (RCA)',
                    'description': 'Conduct systematic RCA using 5 Whys and Fishbone diagram techniques.',
                    'category': 'compliance_action',
                    'urgency': 'IMMEDIATE',
                    'url': 'https://asq.org/quality-resources/root-cause-analysis',
                    'success_probability': 0.85
                },
                {
                    'title': 'Quality Certification Renewal',
                    'description': 'Review and renew quality certifications (ISO, FDA, CE) before expiration.',
                    'category': 'certification_review',
                    'urgency': 'PLANNED',
                    'url': 'https://www.qualitydigest.com/inside/certification',
                    'success_probability': 0.95
                }
            ],
            'DELIVERY': [
                {
                    'title': 'Multi-Carrier Strategy',
                    'description': 'Diversify shipping carriers to reduce single-point-of-failure risks.',
                    'category': 'carrier_switch',
                    'urgency': 'URGENT',
                    'url': 'https://www.logistics.dhl.com/global-en/home/insights-and-innovation.html',
                    'success_probability': 0.87
                },
                {
                    'title': 'Last-Mile Optimization',
                    'description': 'Implement route optimization and local delivery partnerships.',
                    'category': 'route_optimization',
                    'urgency': 'STANDARD',
                    'url': 'https://www.mckinsey.com/industries/travel-logistics-and-infrastructure/our-insights',
                    'success_probability': 0.85
                },
                {
                    'title': 'Customer Delay Communication',
                    'description': 'Proactive customer notification with revised delivery timelines and compensation.',
                    'category': 'delivery_rescheduling',
                    'urgency': 'IMMEDIATE',
                    'url': 'https://hbr.org/topic/customer-service',
                    'success_probability': 0.92
                },
                {
                    'title': 'Protective Packaging Review',
                    'description': 'Assess and upgrade packaging to prevent damage during transit.',
                    'category': 'packaging_redesign',
                    'urgency': 'PLANNED',
                    'url': 'https://www.packagingdigest.com/protective-packaging',
                    'success_probability': 0.88
                }
            ],
            'PRODUCTION': [
                {
                    'title': 'Predictive Maintenance Implementation',
                    'description': 'Deploy IoT sensors and ML models for predictive equipment maintenance.',
                    'category': 'maintenance_schedule',
                    'urgency': 'PLANNED',
                    'url': 'https://www.industryweek.com/technology-and-iiot/article/predictive-maintenance',
                    'success_probability': 0.90
                },
                {
                    'title': 'Emergency Equipment Rental',
                    'description': 'Source temporary equipment from rental providers during repair period.',
                    'category': 'equipment_repair',
                    'urgency': 'IMMEDIATE',
                    'url': 'https://www.plantengineering.com/articles/equipment-rental-strategies',
                    'success_probability': 0.78
                },
                {
                    'title': 'Production Line Rebalancing',
                    'description': 'Redistribute workload across available production lines to maintain output.',
                    'category': 'capacity_adjustment',
                    'urgency': 'URGENT',
                    'url': 'https://www.lean.org/lexicon-terms/line-balancing',
                    'success_probability': 0.85
                },
                {
                    'title': 'Lean Manufacturing Implementation',
                    'description': 'Apply lean principles to eliminate waste and improve workflow efficiency.',
                    'category': 'workflow_optimization',
                    'urgency': 'STANDARD',
                    'url': 'https://www.lean.org/explore-lean/what-is-lean',
                    'success_probability': 0.88
                }
            ],
            'BRAND': [
                {
                    'title': 'Crisis Communication Protocol',
                    'description': 'Activate crisis PR team with prepared messaging and spokesperson briefings.',
                    'category': 'crisis_communication',
                    'urgency': 'IMMEDIATE',
                    'url': 'https://www.prsa.org/intelligence/tactical-resources/crisis-communication',
                    'success_probability': 0.82
                },
                {
                    'title': 'Social Media Monitoring & Response',
                    'description': 'Deploy real-time social listening and rapid response team.',
                    'category': 'pr_campaign',
                    'urgency': 'URGENT',
                    'url': 'https://sproutsocial.com/insights/social-media-crisis-management',
                    'success_probability': 0.80
                },
                {
                    'title': 'Stakeholder Briefing Program',
                    'description': 'Proactive communication to investors, partners, and key stakeholders.',
                    'category': 'stakeholder_outreach',
                    'urgency': 'URGENT',
                    'url': 'https://hbr.org/topic/stakeholder-management',
                    'success_probability': 0.88
                },
                {
                    'title': 'Brand Reputation Recovery Plan',
                    'description': 'Long-term brand rebuilding strategy with measurable milestones.',
                    'category': 'reputation_recovery',
                    'urgency': 'PLANNED',
                    'url': 'https://www.brandingstrategyinsider.com/topic/brand-reputation',
                    'success_probability': 0.85
                }
            ]
        }

        templates = external_solutions.get(risk_type, external_solutions['SUPPLY_CHAIN'])
        solutions = []

        for template in templates:
            solutions.append({
                'title': template['title'],
                'description': template['description'],
                'source': 'grok',
                'source_type': 'External Intelligence',
                'reference': template['url'],
                'reference_type': 'url',
                'urgency': template['urgency'],
                'confidence': 0.75,
                'success_probability': template['success_probability'],
                'solution_category': template['category']
            })

        return solutions

    def _organize_by_tier(self, solutions: List[Dict]) -> List[Dict]:
        """
        Organize solutions into tiers from lowest to highest urgency.
        Tier 4 (PLANNED) → Tier 3 (STANDARD) → Tier 2 (URGENT) → Tier 1 (IMMEDIATE)
        """
        tiers = {
            4: {'urgency': 'PLANNED', 'solutions': []},
            3: {'urgency': 'STANDARD', 'solutions': []},
            2: {'urgency': 'URGENT', 'solutions': []},
            1: {'urgency': 'IMMEDIATE', 'solutions': []}
        }

        for solution in solutions:
            urgency = solution.get('urgency', 'STANDARD')
            tier_num = self.URGENCY_TIER_ORDER.get(urgency, 3)
            tiers[tier_num]['solutions'].append(solution)

        # Sort solutions within each tier by success_probability (descending)
        for tier in tiers.values():
            tier['solutions'].sort(
                key=lambda x: x.get('success_probability', 0),
                reverse=True
            )

        # Return tiers from lowest urgency (4) to highest (1)
        result = []
        for tier_num in [4, 3, 2, 1]:
            if tiers[tier_num]['solutions']:
                result.append({
                    'tier': tier_num,
                    'urgency': tiers[tier_num]['urgency'],
                    'urgency_label': self._get_urgency_label(tiers[tier_num]['urgency']),
                    'solutions': tiers[tier_num]['solutions']
                })

        return result

    def _get_urgency_label(self, urgency: str) -> str:
        """Get human-readable urgency label."""
        labels = {
            'IMMEDIATE': 'Tier 1: IMMEDIATE (Crisis Response)',
            'URGENT': 'Tier 2: URGENT (Priority Action)',
            'STANDARD': 'Tier 3: STANDARD (Normal Operations)',
            'PLANNED': 'Tier 4: PLANNED (Strategic Initiative)'
        }
        return labels.get(urgency, urgency)


# =========================================================================
# PROCUREMENT SUGGESTION ENGINE
# =========================================================================

class ProcurementSuggestionEngine:
    """
    Generates automated procurement/purchasing suggestions based on detected risks.

    When a risk is identified (equipment failure, supply shortage, quality issue),
    this engine suggests:
    - Replacement items/parts
    - Alternative suppliers
    - Inventory replenishment
    - Emergency procurement actions

    Supabase Tables Required:

    CREATE TABLE suppliers (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name TEXT NOT NULL,
        category TEXT,  -- equipment, raw_materials, services, etc.
        risk_types TEXT[],  -- which risk types they can address
        lead_time_days INT,
        reliability_score FLOAT,  -- 0-1
        is_backup BOOLEAN DEFAULT false,
        contact_info JSONB,
        pricing_tier TEXT,  -- budget, standard, premium
        certifications TEXT[],
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE inventory_items (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name TEXT NOT NULL,
        sku TEXT UNIQUE,
        category TEXT,
        risk_types TEXT[],  -- which risk types this item addresses
        current_stock INT,
        reorder_point INT,
        reorder_quantity INT,
        unit_cost FLOAT,
        supplier_id UUID REFERENCES suppliers(id),
        lead_time_days INT,
        is_critical BOOLEAN DEFAULT false,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE procurement_history (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        risk_id UUID,
        item_id UUID REFERENCES inventory_items(id),
        supplier_id UUID REFERENCES suppliers(id),
        quantity INT,
        status TEXT,  -- suggested, approved, ordered, delivered
        suggested_at TIMESTAMPTZ DEFAULT NOW(),
        approved_at TIMESTAMPTZ,
        ordered_at TIMESTAMPTZ,
        delivered_at TIMESTAMPTZ
    );
    """

    # Risk type to procurement category mapping
    RISK_TO_CATEGORY = {
        'SUPPLY_CHAIN': ['raw_materials', 'components', 'logistics'],
        'QUALITY': ['testing_equipment', 'quality_tools', 'certifications'],
        'DELIVERY': ['logistics', 'packaging', 'transportation'],
        'PRODUCTION': ['equipment', 'machinery', 'spare_parts', 'maintenance'],
        'BRAND': ['marketing', 'pr_services', 'communications']
    }

    # Severity to urgency mapping
    SEVERITY_TO_URGENCY = {
        'CRITICAL': {'urgency': 'IMMEDIATE', 'max_lead_days': 1, 'budget_multiplier': 2.0},
        'HIGH': {'urgency': 'URGENT', 'max_lead_days': 3, 'budget_multiplier': 1.5},
        'MEDIUM': {'urgency': 'STANDARD', 'max_lead_days': 7, 'budget_multiplier': 1.0},
        'LOW': {'urgency': 'PLANNED', 'max_lead_days': 30, 'budget_multiplier': 0.9}
    }

    # Keywords that trigger specific procurement suggestions
    PROCUREMENT_KEYWORDS = {
        'equipment': ['machine', 'equipment', 'device', 'tool', 'system', 'hardware'],
        'spare_parts': ['part', 'component', 'replacement', 'spare', 'repair'],
        'raw_materials': ['material', 'supply', 'inventory', 'stock', 'resource'],
        'services': ['service', 'maintenance', 'support', 'consulting', 'expertise'],
        'logistics': ['shipping', 'transport', 'delivery', 'carrier', 'freight'],
        'safety': ['safety', 'protective', 'compliance', 'regulatory', 'certification']
    }

    def __init__(self, supabase_client=None, grok_engine=None):
        self.supabase = supabase_client
        self.grok_engine = grok_engine

    def generate_suggestions(self, risk_type: str, severity: str, description: str,
                             structured_data: Dict = None) -> Dict:  # noqa: ARG002 - structured_data reserved for future use
        """
        Generate procurement suggestions based on risk analysis.

        Returns:
        {
            'has_suggestions': bool,
            'urgency': str,
            'categories': list,
            'items': list,
            'suppliers': list,
            'estimated_cost': float,
            'recommended_actions': list
        }
        """
        urgency_config = self.SEVERITY_TO_URGENCY.get(severity, self.SEVERITY_TO_URGENCY['MEDIUM'])
        categories = self._detect_categories(risk_type, description)

        suggestions = {
            'has_suggestions': False,
            'urgency': urgency_config['urgency'],
            'max_lead_days': urgency_config['max_lead_days'],
            'budget_multiplier': urgency_config['budget_multiplier'],
            'categories': categories,
            'items': [],
            'suppliers': [],
            'estimated_cost': 0.0,
            'recommended_actions': []
        }

        # Get items from database if available
        if self.supabase:
            suggestions['items'] = self._get_suggested_items(categories, urgency_config)
            suggestions['suppliers'] = self._get_suggested_suppliers(categories, urgency_config)

        # Generate AI-powered suggestions if no database items found
        if not suggestions['items'] and self.grok_engine:
            ai_suggestions = self._generate_ai_suggestions(risk_type, severity, description)
            suggestions['ai_recommendations'] = ai_suggestions

        # Generate recommended actions
        suggestions['recommended_actions'] = self._generate_actions(
            risk_type, severity, categories, suggestions['items']
        )

        suggestions['has_suggestions'] = bool(
            suggestions['items'] or
            suggestions['suppliers'] or
            suggestions.get('ai_recommendations') or
            suggestions['recommended_actions']
        )

        # Calculate estimated cost
        suggestions['estimated_cost'] = self._calculate_estimated_cost(
            suggestions['items'],
            urgency_config['budget_multiplier']
        )

        return suggestions

    def _detect_categories(self, risk_type: str, description: str) -> List[str]:
        """Detect procurement categories from risk type and description."""
        categories = set()

        # Add categories based on risk type
        if risk_type in self.RISK_TO_CATEGORY:
            categories.update(self.RISK_TO_CATEGORY[risk_type])

        # Detect additional categories from description
        desc_lower = description.lower()
        for category, keywords in self.PROCUREMENT_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                categories.add(category)

        return list(categories)

    def _get_suggested_items(self, categories: List[str], urgency_config: Dict) -> List[Dict]:
        """Get suggested inventory items from database."""
        if not self.supabase:
            return []

        try:
            items = []
            for category in categories:
                result = self.supabase.table('inventory_items').select(
                    'id, name, sku, category, current_stock, reorder_point, '
                    'reorder_quantity, unit_cost, lead_time_days, is_critical'
                ).eq('category', category).lte(
                    'lead_time_days', urgency_config['max_lead_days']
                ).execute()

                if result.data:
                    for item in result.data:
                        # Check if reorder needed
                        needs_reorder = item.get('current_stock', 0) <= item.get('reorder_point', 0)
                        items.append({
                            **item,
                            'needs_reorder': needs_reorder,
                            'suggested_quantity': item.get('reorder_quantity', 1) if needs_reorder else 0,
                            'urgency_compatible': True
                        })

            return items

        except Exception as e:
            print(f"Error getting suggested items: {e}")
            return []

    def _get_suggested_suppliers(self, categories: List[str], urgency_config: Dict) -> List[Dict]:
        """Get suggested suppliers from database."""
        if not self.supabase:
            return []

        try:
            suppliers = []
            for category in categories:
                result = self.supabase.table('suppliers').select(
                    'id, name, category, lead_time_days, reliability_score, '
                    'is_backup, pricing_tier, certifications'
                ).eq('category', category).lte(
                    'lead_time_days', urgency_config['max_lead_days']
                ).order('reliability_score', desc=True).limit(3).execute()

                if result.data:
                    suppliers.extend(result.data)

            # Sort by reliability and remove duplicates
            seen = set()
            unique_suppliers = []
            for s in sorted(suppliers, key=lambda x: x.get('reliability_score', 0), reverse=True):
                if s['id'] not in seen:
                    seen.add(s['id'])
                    unique_suppliers.append(s)

            return unique_suppliers[:5]

        except Exception as e:
            print(f"Error getting suggested suppliers: {e}")
            return []

    def _generate_ai_suggestions(self, risk_type: str, severity: str, description: str) -> Dict:
        """Generate AI-powered procurement suggestions via Grok."""
        if not self.grok_engine or not self.grok_engine.api_key:
            return {}

        prompt = f"""Based on this business risk, suggest specific procurement actions:

Risk Type: {risk_type}
Severity: {severity}
Description: {description}

Provide JSON with:
{{
    "items_to_purchase": [
        {{"name": "...", "category": "...", "priority": "high/medium/low", "estimated_cost_usd": 0}}
    ],
    "supplier_types_needed": ["..."],
    "immediate_actions": ["..."],
    "preventive_inventory": ["items to stock for future"]
}}"""

        try:
            response = requests.post(
                self.grok_engine.grok_url,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "grok-2",
                    "temperature": 0.3
                },
                headers={"Authorization": f"Bearer {self.grok_engine.api_key}"},
                timeout=15
            )

            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                # Try to parse JSON from response
                import re
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    return json.loads(match.group())

        except Exception as e:
            print(f"AI suggestion error: {e}")

        return {}

    def _generate_actions(self, risk_type: str, severity: str,
                          categories: List[str], items: List[Dict]) -> List[Dict]:
        """Generate recommended procurement actions."""
        _ = risk_type  # Reserved for risk-type-specific actions in future
        actions = []

        # Critical items needing reorder
        critical_items = [i for i in items if i.get('is_critical') and i.get('needs_reorder')]
        if critical_items:
            actions.append({
                'priority': 'CRITICAL',
                'action': 'EMERGENCY_REORDER',
                'description': f"Immediately reorder {len(critical_items)} critical items",
                'items': [i['name'] for i in critical_items[:3]]
            })

        # Severity-based actions
        if severity == 'CRITICAL':
            actions.append({
                'priority': 'CRITICAL',
                'action': 'ACTIVATE_BACKUP_SUPPLIERS',
                'description': 'Contact backup suppliers for emergency procurement'
            })
            actions.append({
                'priority': 'HIGH',
                'action': 'EXPEDITE_SHIPPING',
                'description': 'Authorize expedited shipping for all related orders'
            })

        elif severity == 'HIGH':
            actions.append({
                'priority': 'HIGH',
                'action': 'REVIEW_INVENTORY',
                'description': f"Review inventory levels for categories: {', '.join(categories[:3])}"
            })

        # Category-specific actions
        if 'spare_parts' in categories or 'equipment' in categories:
            actions.append({
                'priority': 'MEDIUM',
                'action': 'MAINTENANCE_CHECK',
                'description': 'Schedule preventive maintenance check'
            })

        if 'raw_materials' in categories:
            actions.append({
                'priority': 'MEDIUM',
                'action': 'SUPPLIER_DIVERSIFICATION',
                'description': 'Evaluate alternative suppliers to reduce single-source risk'
            })

        return actions

    def _calculate_estimated_cost(self, items: List[Dict], budget_multiplier: float) -> float:
        """Calculate estimated procurement cost."""
        total = 0.0
        for item in items:
            if item.get('needs_reorder'):
                quantity = item.get('suggested_quantity', 1)
                unit_cost = item.get('unit_cost', 0)
                total += quantity * unit_cost

        return round(total * budget_multiplier, 2)

    def record_suggestion(self, risk_id: str, item_id: str, supplier_id: str,
                          quantity: int) -> Optional[str]:
        """Record a procurement suggestion in the database."""
        if not self.supabase:
            return None

        try:
            result = self.supabase.table('procurement_history').insert({
                'risk_id': risk_id,
                'item_id': item_id,
                'supplier_id': supplier_id,
                'quantity': quantity,
                'status': 'suggested'
            }).execute()

            if result.data:
                return result.data[0].get('id')

        except Exception as e:
            print(f"Error recording suggestion: {e}")

        return None

    def update_suggestion_status(self, suggestion_id: str, status: str) -> bool:
        """Update procurement suggestion status."""
        if not self.supabase:
            return False

        try:
            updates = {'status': status}

            if status == 'approved':
                updates['approved_at'] = datetime.now().isoformat()
            elif status == 'ordered':
                updates['ordered_at'] = datetime.now().isoformat()
            elif status == 'delivered':
                updates['delivered_at'] = datetime.now().isoformat()

            self.supabase.table('procurement_history').update(updates).eq(
                'id', suggestion_id
            ).execute()
            return True

        except Exception as e:
            print(f"Error updating suggestion: {e}")
            return False


# =========================================================================
# SCHEMA VALIDATOR
# =========================================================================

class SchemaValidator:
    """Validate schema for all data types."""

    MIN_LENGTH = 20
    MAX_LENGTH = 50000  # Increased: chunking now handles long text

    def validate(self, description: str) -> Tuple[bool, str]:
        """Validate description."""

        if not description or len(description) < self.MIN_LENGTH:
            return False, "Description too short (min 20 characters)"

        if len(description) > self.MAX_LENGTH:
            return False, f"Description too long (max {self.MAX_LENGTH} characters)"

        return True, "Valid"


# =========================================================================
# DOCUMENT STORE (In-Memory Fallback)
# =========================================================================

class DocumentStore:
    """In-memory store - fallback when Supabase unavailable."""

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
# SUPABASE PERSISTENT STORE (with pgvector)
# =========================================================================

class SupabaseRiskStore:
    """
    Persistent storage for risks using Supabase + pgvector.

    Table Schema (risks):
        id: uuid (primary key)
        description: text
        risk_type: text
        severity: text
        confidence: float
        embedding: vector(768)
        structured_data: jsonb
        created_at: timestamptz
        updated_at: timestamptz

    Requires pgvector extension enabled in Supabase:
        CREATE EXTENSION IF NOT EXISTS vector;

    Create table SQL:
        CREATE TABLE risks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            description TEXT NOT NULL,
            risk_type TEXT,
            severity TEXT,
            confidence FLOAT,
            embedding vector(768),
            structured_data JSONB DEFAULT '{}',
            data_type TEXT DEFAULT 'unstructured',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX ON risks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """

    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.table_name = 'risks'

    def store_risk(self, description: str, risk_type: str, severity: str,
                   confidence: float, embedding: np.ndarray,
                   structured_data: Dict = None, data_type: str = 'unstructured') -> Optional[str]:
        """
        Store a risk document with its embedding in Supabase.

        Returns: UUID of stored document, or None on failure
        """
        if not self.supabase:
            return None

        try:
            # Convert numpy array to list for JSON serialization
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

            data = {
                'description': description,
                'risk_type': risk_type,
                'severity': severity,
                'confidence': confidence,
                'embedding': embedding_list,
                'structured_data': structured_data or {},
                'data_type': data_type,
                'updated_at': datetime.now().isoformat()
            }

            result = self.supabase.table(self.table_name).insert(data).execute()

            if result.data and len(result.data) > 0:
                return result.data[0].get('id')

            return None

        except Exception as e:
            print(f"Supabase store error: {e}")
            return None

    def get_risk(self, risk_id: str) -> Optional[Dict]:
        """Retrieve a risk by ID."""
        if not self.supabase:
            return None

        try:
            result = self.supabase.table(self.table_name).select('*').eq('id', risk_id).execute()

            if result.data and len(result.data) > 0:
                return result.data[0]

            return None

        except Exception as e:
            print(f"Supabase get error: {e}")
            return None

    def vector_search(self, query_embedding: np.ndarray, top_k: int = 5,
                      filter_risk_type: str = None, filter_severity: str = None) -> List[Dict]:
        """
        Semantic search using pgvector cosine similarity.

        Uses Supabase RPC function for vector search:

        CREATE OR REPLACE FUNCTION match_risks(
            query_embedding vector(768),
            match_threshold float DEFAULT 0.5,
            match_count int DEFAULT 5,
            filter_risk_type text DEFAULT NULL,
            filter_severity text DEFAULT NULL
        )
        RETURNS TABLE (
            id uuid,
            description text,
            risk_type text,
            severity text,
            confidence float,
            structured_data jsonb,
            similarity float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                r.id,
                r.description,
                r.risk_type,
                r.severity,
                r.confidence,
                r.structured_data,
                1 - (r.embedding <=> query_embedding) AS similarity
            FROM risks r
            WHERE
                (filter_risk_type IS NULL OR r.risk_type = filter_risk_type)
                AND (filter_severity IS NULL OR r.severity = filter_severity)
                AND 1 - (r.embedding <=> query_embedding) > match_threshold
            ORDER BY r.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
        """
        if not self.supabase:
            return []

        try:
            embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

            result = self.supabase.rpc('match_risks', {
                'query_embedding': embedding_list,
                'match_threshold': 0.5,
                'match_count': top_k,
                'filter_risk_type': filter_risk_type,
                'filter_severity': filter_severity
            }).execute()

            return result.data if result.data else []

        except Exception as e:
            print(f"Supabase vector search error: {e}")
            # Fallback: try simple query without vector search
            return self._fallback_search(top_k, filter_risk_type, filter_severity)

    def _fallback_search(self, top_k: int, filter_risk_type: str = None,
                         filter_severity: str = None) -> List[Dict]:
        """Fallback search without vector similarity (if RPC not available)."""
        if not self.supabase:
            return []

        try:
            query = self.supabase.table(self.table_name).select(
                'id, description, risk_type, severity, confidence, structured_data, created_at'
            )

            if filter_risk_type:
                query = query.eq('risk_type', filter_risk_type)

            if filter_severity:
                query = query.eq('severity', filter_severity)

            result = query.order('created_at', desc=True).limit(top_k).execute()

            return result.data if result.data else []

        except Exception as e:
            print(f"Supabase fallback search error: {e}")
            return []

    def get_recent_risks(self, limit: int = 10) -> List[Dict]:
        """Get most recent risks."""
        if not self.supabase:
            return []

        try:
            result = self.supabase.table(self.table_name).select(
                'id, description, risk_type, severity, confidence, structured_data, created_at'
            ).order('created_at', desc=True).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            print(f"Supabase get recent error: {e}")
            return []

    def get_risks_by_type(self, risk_type: str, limit: int = 10) -> List[Dict]:
        """Get risks filtered by type."""
        if not self.supabase:
            return []

        try:
            result = self.supabase.table(self.table_name).select('*').eq(
                'risk_type', risk_type
            ).order('created_at', desc=True).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            print(f"Supabase get by type error: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        if not self.supabase:
            return {'connected': False}

        try:
            # Count total
            total_result = self.supabase.table(self.table_name).select('id', count='exact').execute()

            # Count by risk type
            type_counts = {}
            for risk_type in ['SUPPLY_CHAIN', 'QUALITY', 'DELIVERY', 'PRODUCTION', 'BRAND']:
                type_result = self.supabase.table(self.table_name).select(
                    'id', count='exact'
                ).eq('risk_type', risk_type).execute()
                type_counts[risk_type] = type_result.count if type_result.count else 0

            # Count by severity
            severity_counts = {}
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                sev_result = self.supabase.table(self.table_name).select(
                    'id', count='exact'
                ).eq('severity', severity).execute()
                severity_counts[severity] = sev_result.count if sev_result.count else 0

            return {
                'connected': True,
                'total_documents': total_result.count if total_result.count else 0,
                'by_risk_type': type_counts,
                'by_severity': severity_counts,
                'storage_type': 'supabase_pgvector'
            }

        except Exception as e:
            print(f"Supabase stats error: {e}")
            return {'connected': False, 'error': str(e)}

    def delete_risk(self, risk_id: str) -> bool:
        """Delete a risk by ID."""
        if not self.supabase:
            return False

        try:
            self.supabase.table(self.table_name).delete().eq('id', risk_id).execute()
            return True

        except Exception as e:
            print(f"Supabase delete error: {e}")
            return False

    def update_risk(self, risk_id: str, updates: Dict) -> bool:
        """Update a risk document."""
        if not self.supabase:
            return False

        try:
            updates['updated_at'] = datetime.now().isoformat()
            self.supabase.table(self.table_name).update(updates).eq('id', risk_id).execute()
            return True

        except Exception as e:
            print(f"Supabase update error: {e}")
            return False


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
    Grok Intelligence synthesizes findings and provides industry context.

    Input Handling:
    - Accepts str, dict, or JSON string
    - Handles long text via chunking (beyond 512 tokens)
    - Uses ML-based classification via DistilBERT embeddings
    """

    # Risk type centroids for ML-based classification
    RISK_TYPE_EXAMPLES = {
        'SUPPLY_CHAIN': "supplier vendor sourcing procurement logistics supply chain disruption shortage",
        'QUALITY': "defect specification compliance quality control inspection failure standards",
        'DELIVERY': "shipment logistics delay transportation shipping late delivery carrier",
        'PRODUCTION': "equipment manufacturing capacity production line downtime maintenance",
        'BRAND': "reputation customer perception brand image public relations media"
    }

    SEVERITY_EXAMPLES = {
        'CRITICAL': "crisis bankruptcy catastrophic shutdown complete failure total loss emergency disaster",
        'HIGH': "urgent significant impact revenue major serious substantial damage",
        'MEDIUM': "moderate issue problem concern noticeable affecting",
        'LOW': "minor small negligible minimal slight inconvenience"
    }

    def __init__(self, grok_api_key: str = None, supabase_client=None):
        super().__init__()

        # Load DistilBERT (unified embedding for all data types)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.embedding_dim = 768
        self.max_tokens = 512  # DistilBERT limit
        self.chunk_overlap = 50  # Token overlap between chunks

        # Components
        self.grok_engine = GrokIntelligenceEngine(api_key=grok_api_key)
        self.correlation_engine = RiskCorrelationEngine()
        self.schema_validator = SchemaValidator()
        self.document_store = DocumentStore()  # In-memory fallback
        self.supabase = supabase_client

        # Persistent storage with pgvector (if Supabase available)
        self.persistent_store = SupabaseRiskStore(supabase_client) if supabase_client else None
        self.use_persistent_storage = supabase_client is not None

        # Procurement suggestion engine
        self.procurement_engine = ProcurementSuggestionEngine(
            supabase_client=supabase_client,
            grok_engine=self.grok_engine
        )

        # Dual-source solution engine (Supabase + Grok simultaneously)
        self.solution_engine = DualSourceSolutionEngine(
            supabase_client=supabase_client,
            grok_engine=self.grok_engine
        )

        # In-memory vector store (fallback + cache)
        self.vector_store = []

        # Pre-compute classification embeddings for ML-based classification
        self._risk_type_embeddings = {}
        self._severity_embeddings = {}
        self._initialize_classification_embeddings()

    def _initialize_classification_embeddings(self):
        """Pre-compute embeddings for ML-based classification."""
        for risk_type, example_text in self.RISK_TYPE_EXAMPLES.items():
            self._risk_type_embeddings[risk_type] = self._embed_single_chunk(example_text)

        for severity, example_text in self.SEVERITY_EXAMPLES.items():
            self._severity_embeddings[severity] = self._embed_single_chunk(example_text)

    def _embed_single_chunk(self, text: str) -> np.ndarray:
        """Embed a single chunk of text (internal, no chunking)."""
        inputs = self.tokenizer.encode(text, return_tensors='pt', max_length=self.max_tokens, truncation=True)

        with torch.no_grad():
            outputs = self.distilbert(inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.numpy()[0]

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split long text into overlapping chunks for processing.

        Handles text longer than 512 tokens by:
        1. Tokenizing the full text
        2. Splitting into chunks with overlap
        3. Decoding back to text chunks
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.max_tokens - 2:  # -2 for [CLS] and [SEP]
            return [text]

        chunks = []
        stride = self.max_tokens - self.chunk_overlap - 2

        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + self.max_tokens - 2]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            if i + self.max_tokens - 2 >= len(tokens):
                break

        return chunks

    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert any text to 768-dim embedding via DistilBERT.

        For long text (>512 tokens):
        - Splits into overlapping chunks
        - Embeds each chunk
        - Returns weighted average (later chunks weighted less)
        """
        chunks = self._chunk_text(text)

        if len(chunks) == 1:
            return self._embed_single_chunk(chunks[0])

        # Embed all chunks with diminishing weights
        embeddings = []
        weights = []
        for i, chunk in enumerate(chunks):
            emb = self._embed_single_chunk(chunk)
            embeddings.append(emb)
            weights.append(1.0 / (i + 1))  # First chunk weighted highest

        # Weighted average
        weights = np.array(weights) / sum(weights)
        combined = np.zeros(self.embedding_dim)
        for emb, weight in zip(embeddings, weights):
            combined += emb * weight

        return combined

    def _parse_input(self, data: Union[str, Dict]) -> Tuple[str, Dict]:
        """
        Parse input data into description and structured_data.

        Accepts:
        - str: Plain text description
        - dict: Structured input with 'description' and optional fields
        - str (JSON): JSON string that will be parsed to dict

        Returns: (description, structured_data)
        """
        structured_data = {}

        # Handle string input
        if isinstance(data, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    data = parsed
                else:
                    return data.strip(), structured_data
            except (json.JSONDecodeError, TypeError):
                return data.strip(), structured_data

        # Handle dict input
        if isinstance(data, dict):
            description = data.get('description', '')

            # If no description field, try to construct from other fields
            if not description:
                # Try common field names
                for field in ['text', 'content', 'message', 'risk', 'input']:
                    if field in data and data[field]:
                        description = str(data[field])
                        break

            # If still no description, serialize the whole dict
            if not description:
                description = json.dumps(data, default=str)

            # Extract structured fields
            for key in ['risk_type', 'severity', 'category', 'priority', 'source',
                        'department', 'location', 'impact', 'likelihood', 'owner',
                        'due_date', 'tags', 'metadata', 'attributes']:
                if key in data and data[key] is not None:
                    structured_data[key] = data[key]

            return description.strip(), structured_data

        # Fallback: convert to string
        return str(data).strip(), structured_data
    
    def forward(self, data: Union[str, Dict], analyze_interconnections: bool = True,
                structured_data: Dict = None) -> Dict:
        """
        Analyze risk across all data type paths.

        Args:
            data: Input data - can be:
                - str: Plain text description
                - dict: Structured input with 'description' and optional fields
                - str (JSON): JSON string that will be parsed
            analyze_interconnections: Whether to analyze interconnections
            structured_data: Optional pre-parsed structured data (if already extracted)

        Steps:
        1. Parse input (str, dict, or JSON)
        2. Embed all data types to unified space (with chunking for long text)
        3. Detect conflicts and patterns
        4. Find historical precedents
        5. Process with Grok Intelligence (Agentic RAG) - includes structured_data
        6. ML-based classification via embedding similarity
        7. Generate alerts and solutions
        """

        # Parse input to extract description and structured_data
        description, parsed_structured = self._parse_input(data)

        # Merge with explicitly passed structured_data
        if structured_data:
            parsed_structured.update(structured_data)

        # Validate description
        valid, msg = self.schema_validator.validate(description)
        if not valid:
            return {'success': False, 'error': msg}

        # Embed to unified space (handles long text via chunking)
        embedding = self.embed_text(description)

        # Track chunking info
        chunks = self._chunk_text(description)
        chunking_info = {
            'was_chunked': len(chunks) > 1,
            'num_chunks': len(chunks),
            'original_length': len(description)
        }

        # Detect issues
        self_conflicts = self.correlation_engine.detect_self_conflicts(description)

        # Find historical matches
        historical_matches = self.correlation_engine.find_historical_matches(embedding, self.vector_store)

        # Grok Intelligence (Agentic RAG) - now includes structured_data
        grok_findings = self.grok_engine.agentic_rag_process(description, parsed_structured)

        # ML-based classification via embedding similarity
        risk_type = self._classify_risk_ml(embedding, parsed_structured)
        severity = self._classify_severity_ml(embedding, parsed_structured)
        confidence = self._calculate_confidence(historical_matches, grok_findings)

        # Generate procurement suggestions based on risk
        procurement_suggestions = self.procurement_engine.generate_suggestions(
            risk_type=risk_type,
            severity=severity,
            description=description,
            structured_data=parsed_structured
        )

        # Store in persistent storage (Supabase + pgvector) if available
        doc_id = None
        storage_type = 'in_memory'

        if self.use_persistent_storage and self.persistent_store:
            doc_id = self.persistent_store.store_risk(
                description=description,
                risk_type=risk_type,
                severity=severity,
                confidence=confidence,
                embedding=embedding,
                structured_data=parsed_structured,
                data_type='unified'
            )
            if doc_id:
                storage_type = 'supabase_pgvector'

        # Fallback to in-memory storage
        if not doc_id:
            doc_id = self.document_store.store(description, {
                'risk_type': risk_type,
                'severity': severity,
                'data_type': 'unified',
                'structured_data': parsed_structured
            })
            storage_type = 'in_memory'

        # Also cache in memory for fast access
        self.vector_store.append({
            'doc_id': doc_id,
            'description': description,
            'embedding': embedding,
            'risk_type': risk_type,
            'severity': severity,
            'structured_data': parsed_structured,
            'created_at': datetime.now().isoformat()
        })

        return {
            'success': True,
            'risk_type': risk_type,
            'severity': severity,
            'confidence': confidence,
            'doc_id': doc_id,
            'storage_type': storage_type,
            'chunking_info': chunking_info,
            'structured_data': parsed_structured,
            'analysis': {
                'self_conflicts': self_conflicts,
                'historical_matches': historical_matches,
                'grok_intelligence': grok_findings
            },
            'procurement': procurement_suggestions
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
    
    def semantic_search(self, query: str, top_k: int = 5,
                        filter_risk_type: str = None, filter_severity: str = None) -> List[Dict]:
        """
        Semantic search across stored vectors.

        Uses pgvector for persistent storage, falls back to in-memory.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_risk_type: Optional filter by risk type
            filter_severity: Optional filter by severity
        """
        query_embedding = self.embed_text(query)

        # Try pgvector search first (persistent storage)
        if self.use_persistent_storage and self.persistent_store:
            pg_results = self.persistent_store.vector_search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_risk_type=filter_risk_type,
                filter_severity=filter_severity
            )

            if pg_results:
                return [{
                    'doc_id': r.get('id'),
                    'similarity': float(r.get('similarity', 0)),
                    'description': r.get('description', '')[:200],
                    'risk_type': r.get('risk_type'),
                    'severity': r.get('severity'),
                    'confidence': r.get('confidence'),
                    'structured_data': r.get('structured_data', {}),
                    'source': 'supabase_pgvector'
                } for r in pg_results]

        # Fallback to in-memory search
        results = []
        for doc in self.vector_store:
            # Apply filters
            if filter_risk_type and doc.get('risk_type') != filter_risk_type:
                continue
            if filter_severity and doc.get('severity') != filter_severity:
                continue

            similarity = np.dot(query_embedding, doc['embedding']) / (
                norm(query_embedding) * norm(doc['embedding']) + 1e-10
            )
            results.append({
                'doc_id': doc['doc_id'],
                'similarity': float(similarity),
                'description': doc['description'][:200],
                'risk_type': doc['risk_type'],
                'severity': doc.get('severity'),
                'structured_data': doc.get('structured_data', {}),
                'source': 'in_memory'
            })

        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    def get_vector_store_stats(self) -> Dict:
        """Get statistics about stored vectors."""

        stats = {
            'in_memory_documents': len(self.vector_store),
            'embedding_dimension': self.embedding_dim,
            'embedding_model': 'DistilBERT',
            'data_types_supported': ['structured', 'semi-structured', 'unstructured'],
            'persistent_storage_enabled': self.use_persistent_storage
        }

        # Add Supabase/pgvector stats if available
        if self.use_persistent_storage and self.persistent_store:
            pg_stats = self.persistent_store.get_stats()
            stats['persistent_storage'] = pg_stats
            stats['total_documents'] = pg_stats.get('total_documents', 0) + len(self.vector_store)
        else:
            stats['total_documents'] = len(self.vector_store)
            stats['persistent_storage'] = {'connected': False}

        return stats

    def get_risk(self, risk_id: str) -> Optional[Dict]:
        """Retrieve a risk by ID from persistent or in-memory storage."""

        # Try persistent storage first
        if self.use_persistent_storage and self.persistent_store:
            result = self.persistent_store.get_risk(risk_id)
            if result:
                result['source'] = 'supabase_pgvector'
                return result

        # Fallback to in-memory
        doc = self.document_store.retrieve(risk_id)
        if doc:
            doc['source'] = 'in_memory'
            return doc

        # Check vector store cache
        for item in self.vector_store:
            if item.get('doc_id') == risk_id:
                return {**item, 'source': 'in_memory_cache'}

        return None

    def get_recent_risks(self, limit: int = 10) -> List[Dict]:
        """Get most recent risks from storage."""

        if self.use_persistent_storage and self.persistent_store:
            results = self.persistent_store.get_recent_risks(limit)
            if results:
                return [{'source': 'supabase_pgvector', **r} for r in results]

        # Fallback to in-memory (sorted by created_at desc)
        sorted_store = sorted(
            self.vector_store,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )
        return [{'source': 'in_memory', **item} for item in sorted_store[:limit]]

    def delete_risk(self, risk_id: str) -> bool:
        """Delete a risk from storage."""

        deleted = False

        # Delete from persistent storage
        if self.use_persistent_storage and self.persistent_store:
            deleted = self.persistent_store.delete_risk(risk_id)

        # Also remove from in-memory cache
        self.vector_store = [v for v in self.vector_store if v.get('doc_id') != risk_id]

        # Remove from document store
        if risk_id in self.document_store.documents:
            del self.document_store.documents[risk_id]
            deleted = True

        return deleted
    
    def _classify_risk_ml(self, embedding: np.ndarray, structured_data: Dict = None) -> str:
        """
        ML-based risk classification using embedding similarity.

        Uses cosine similarity between input embedding and pre-computed
        risk type embeddings to find the best match.

        Falls back to structured_data['risk_type'] if provided and valid.
        """
        # If structured data provides a valid risk_type, use it
        if structured_data and 'risk_type' in structured_data:
            provided = structured_data['risk_type']
            if provided in self.RISK_TYPE_EXAMPLES:
                return provided

        # ML-based classification via embedding similarity
        best_type = 'SUPPLY_CHAIN'
        best_similarity = -1.0

        for risk_type, type_embedding in self._risk_type_embeddings.items():
            similarity = np.dot(embedding, type_embedding) / (
                norm(embedding) * norm(type_embedding) + 1e-10
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_type = risk_type

        return best_type

    def _classify_severity_ml(self, embedding: np.ndarray, structured_data: Dict = None) -> str:
        """
        ML-based severity classification using embedding similarity.

        Uses cosine similarity between input embedding and pre-computed
        severity embeddings to find the best match.

        Falls back to structured_data['severity'] if provided and valid.
        """
        # If structured data provides a valid severity, use it
        if structured_data and 'severity' in structured_data:
            provided = structured_data['severity']
            if provided in self.SEVERITY_EXAMPLES:
                return provided

        # ML-based classification via embedding similarity
        best_severity = 'MEDIUM'
        best_similarity = -1.0

        for severity, severity_embedding in self._severity_embeddings.items():
            similarity = np.dot(embedding, severity_embedding) / (
                norm(embedding) * norm(severity_embedding) + 1e-10
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_severity = severity

        return best_severity

    def _classify_risk(self, description: str) -> str:
        """Legacy keyword-based classification (kept for backward compatibility)."""
        embedding = self.embed_text(description)
        return self._classify_risk_ml(embedding)

    def _classify_severity(self, description: str) -> str:
        """Legacy keyword-based classification (kept for backward compatibility)."""
        embedding = self.embed_text(description)
        return self._classify_severity_ml(embedding)
    
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

def create_dual_path_transformer(
    grok_api_key: str = None,
    supabase_client=None,
    client_slug: str = None,
    api_key: str = None
) -> DualPathRiskTransformer:
    """
    Create and initialize the transformer.

    Multi-tenant usage:
        # Option 1: Direct client slug
        transformer = create_dual_path_transformer(client_slug='acme-corp')

        # Option 2: API key authentication
        transformer = create_dual_path_transformer(api_key='vgl_prod_xxx...')

        # Option 3: Direct Supabase client (legacy/single-tenant)
        transformer = create_dual_path_transformer(supabase_client=my_client)

    Args:
        grok_api_key: X.AI API key for Grok. If None, uses XAI_API_KEY env var.
        supabase_client: Direct Supabase client (bypasses multi-tenant).
        client_slug: Client identifier to look up in master database.
        api_key: Vigil API key to authenticate and get client connection.

    Returns:
        Configured DualPathRiskTransformer instance.
    """
    # Get Grok API key from env if not provided
    if not grok_api_key:
        grok_api_key = os.getenv('XAI_API_KEY')

    # Determine Supabase client to use
    effective_client = supabase_client

    if not effective_client and api_key:
        # Authenticate via API key
        effective_client, client_info = client_manager.get_client_by_api_key(api_key)
        if client_info:
            print(f"Authenticated as client: {client_info.get('name')}")

    if not effective_client and client_slug:
        # Look up by slug
        effective_client = client_manager.get_client_by_slug(client_slug)
        if effective_client:
            print(f"Connected to client: {client_slug}")

    if not effective_client:
        # Check for default client in env
        default_slug = os.getenv('DEFAULT_CLIENT_SLUG')
        if default_slug:
            effective_client = client_manager.get_client_by_slug(default_slug)
            if effective_client:
                print(f"Using default client: {default_slug}")

    return DualPathRiskTransformer(
        grok_api_key=grok_api_key,
        supabase_client=effective_client
    )


def create_transformer_for_request(api_key: str) -> Tuple[Optional[DualPathRiskTransformer], Optional[Dict]]:
    """
    Create a transformer for an API request.

    This is the primary entry point for handling multi-tenant API requests.
    Validates the API key, gets the client connection, and returns both
    the transformer and client info.

    Args:
        api_key: The Vigil API key from the request header.

    Returns:
        Tuple of (transformer, client_info) or (None, None) if invalid.

    Example:
        transformer, client = create_transformer_for_request(request.headers.get('X-API-Key'))
        if not transformer:
            return {"error": "Invalid API key"}, 401

        result = transformer.analyze_risk("Supply chain delay...")
    """
    client, client_info = client_manager.get_client_by_api_key(api_key)

    if not client or not client_info:
        return None, None

    grok_api_key = os.getenv('XAI_API_KEY')

    transformer = DualPathRiskTransformer(
        grok_api_key=grok_api_key,
        supabase_client=client
    )

    # Record API call
    client_manager.record_usage(
        client_id=client_info.get('id'),
        api_calls=1
    )

    return transformer, client_info