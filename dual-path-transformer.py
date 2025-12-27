"""
Dual-Path Risk Management Transformer with Vector Storage, Query Search & Notifications
Combines Structured Data Transformer (PostgreSQL) with Unstructured Data Transformer
Includes vector storage, semantic search, notification system, schema validation, and human-friendly outputs
Compatible with PostgreSQL pgvector extension
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sentence_transformers import models
import json
import re
from typing import Dict, List, Union, Optional, Tuple
import numpy as np
from numpy.linalg import norm
from collections import defaultdict
from datetime import datetime


# ===== SCHEMA VALIDATOR (Built-in) =====

class SchemaValidator:
    """
    Schema validation for both write and read operations
    Enforces data integrity at both ends
    """
    
    # ===== SCHEMA ON WRITE =====
    STRUCTURED_SCHEMA = {
        'incident_id': {'type': str, 'required': True, 'description': 'Incident ID'},
        'product_id': {'type': str, 'required': False, 'description': 'Product ID'},
        'risk_category': {
            'type': str, 'required': True,
            'allowed_values': ['Product Risk', 'Service Risk', 'Brand Risk', 'Supply Chain', 'Quality Control'],
            'description': 'Risk category'
        },
        'severity': {
            'type': str, 'required': True,
            'allowed_values': ['Low', 'Medium', 'High', 'Critical'],
            'description': 'Severity level'
        },
        'region': {
            'type': str, 'required': False,
            'allowed_values': ['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania'],
            'description': 'Geographic region'
        },
        'defect_rate': {'type': float, 'required': False, 'min': 0.0, 'max': 1.0, 'description': 'Defect rate'},
        'description': {'type': str, 'required': False, 'min_length': 10, 'max_length': 500, 'description': 'Description'}
    }
    
    # ===== SCHEMA ON READ =====
    OUTPUT_SCHEMA = {
        'doc_id': {'type': int, 'required': True, 'description': 'Document ID'},
        'embedding': {'type': list, 'required': True, 'length': 768, 'element_type': float, 'description': 'Embedding'},
        'data_source': {'type': str, 'required': True, 'allowed_values': ['postgresql', 'unstructured'], 'description': 'Source'},
        'metadata': {'type': dict, 'required': True, 'required_fields': ['category', 'severity', 'confidence'], 'description': 'Metadata'},
        'category': {'type': str, 'required': True, 'allowed_values': ['Product Risk', 'Service Risk', 'Brand Risk'], 'description': 'Category'},
        'severity': {'type': str, 'required': True, 'allowed_values': ['Low', 'Medium', 'High', 'Critical'], 'description': 'Severity'},
        'confidence': {'type': float, 'required': True, 'min': 0.0, 'max': 1.0, 'description': 'Confidence'}
    }
    
    ALERT_SCHEMA = {
        'doc_id': {'type': int, 'required': True, 'description': 'Document ID'},
        'timestamp': {'type': str, 'required': True, 'description': 'Timestamp'},
        'severity': {'type': str, 'required': True, 'allowed_values': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], 'description': 'Severity'},
        'category': {'type': str, 'required': True, 'allowed_values': ['Product Risk', 'Service Risk', 'Brand Risk'], 'description': 'Category'},
        'confidence': {'type': float, 'required': True, 'min': 0.3, 'max': 1.0, 'description': 'Confidence'},
        'message': {'type': str, 'required': True, 'min_length': 20, 'description': 'Message'}
    }
    
    @staticmethod
    def validate_schema_on_write(data: Union[str, Dict], data_type: str) -> tuple:
        """SCHEMA ON WRITE: Validate data entering system"""
        errors = []
        
        if data_type == 'structured':
            if not isinstance(data, dict):
                return False, ["Structured data must be dict"]
            
            for field, rules in SchemaValidator.STRUCTURED_SCHEMA.items():
                if rules.get('required', False) and field not in data:
                    errors.append(f"Required field '{field}' missing")
                elif field in data:
                    value = data[field]
                    if not isinstance(value, rules['type']):
                        errors.append(f"Field '{field}' wrong type")
                    if 'allowed_values' in rules and value not in rules['allowed_values']:
                        errors.append(f"Field '{field}' invalid value")
                    if 'min' in rules and isinstance(value, (int, float)) and value < rules['min']:
                        errors.append(f"Field '{field}' below minimum")
                    if 'max' in rules and isinstance(value, (int, float)) and value > rules['max']:
                        errors.append(f"Field '{field}' exceeds maximum")
        
        elif data_type == 'unstructured':
            if not isinstance(data, str):
                return False, ["Unstructured data must be string"]
            if len(data) < 20:
                errors.append("Content too short (min: **20** chars)")
            if len(data) > 5000:
                errors.append("Content too long (max: **5000** chars)")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_schema_on_read(data: Dict, schema_type: str = 'output') -> tuple:
        """SCHEMA ON READ: Validate data exiting system"""
        errors = []
        schema = SchemaValidator.OUTPUT_SCHEMA if schema_type == 'output' else SchemaValidator.ALERT_SCHEMA
        
        for field, rules in schema.items():
            if rules.get('required', False) and field not in data:
                errors.append(f"Required output field '{field}' missing")
        
        return len(errors) == 0, errors


class NotificationSystem:
    """
    Notification system for risk alerts.
    Tracks triggered alerts, severity levels, and notification history.
    """
    
    def __init__(self):
        self.notifications = []  # List of all notifications
        self.alert_history = defaultdict(list)  # Alert history by doc_id
        self.severity_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.75,
            'critical': 0.9
        }
        self.notification_templates = {
            'critical': "[CRITICAL ALERT] {title} - {description} [Confidence: {confidence}%]",
            'high': "[HIGH PRIORITY] {title} - {description} [Confidence: {confidence}%]",
            'medium': "[MEDIUM ALERT] {title} - {description} [Confidence: {confidence}%]",
            'low': "[INFO] {title} - {description} [Confidence: {confidence}%]"
        }
    
    def get_severity_level(self, confidence: float) -> str:
        """Determine severity level from confidence score."""
        if confidence >= self.severity_thresholds['critical']:
            return 'critical'
        elif confidence >= self.severity_thresholds['high']:
            return 'high'
        elif confidence >= self.severity_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def create_notification(
        self,
        doc_id: int,
        title: str,
        description: str,
        confidence: float,
        risk_category: str,
        severity_level: str,
        data_source: str
    ) -> Dict:
        """
        Create a notification alert.
        
        Returns:
            Notification dictionary
        """
        severity = self.get_severity_level(confidence)
        
        notification = {
            'timestamp': datetime.now().isoformat(),
            'doc_id': doc_id,
            'title': title,
            'description': description,
            'confidence': confidence,
            'confidence_percentage': int(confidence * 100),
            'risk_category': risk_category,
            'severity_level': severity_level,
            'severity_rating': severity.upper(),
            'data_source': data_source,
            'message': self.notification_templates[severity].format(
                title=title,
                description=description,
                confidence=int(confidence * 100)
            )
        }
        
        self.notifications.append(notification)
        self.alert_history[doc_id].append(notification)
        
        return notification
    
    def check_threshold_breach(
        self,
        confidence: float,
        threshold: str = 'medium'
    ) -> bool:
        """Check if confidence breaches severity threshold."""
        return confidence >= self.severity_thresholds.get(threshold, 0.6)
    
    def get_notifications_summary(self) -> str:
        """Get human-friendly summary of all notifications."""
        if not self.notifications:
            return "No alerts at this time."
        
        critical_count = len([n for n in self.notifications if n['severity_rating'] == 'CRITICAL'])
        high_count = len([n for n in self.notifications if n['severity_rating'] == 'HIGH'])
        medium_count = len([n for n in self.notifications if n['severity_rating'] == 'MEDIUM'])
        
        summary = f"\nALERT SUMMARY\n"
        summary += f"Total Alerts: **{len(self.notifications)}**\n"
        if critical_count > 0:
            summary += f"Critical: **{critical_count}**\n"
        if high_count > 0:
            summary += f"High Priority: **{high_count}**\n"
        if medium_count > 0:
            summary += f"Medium: **{medium_count}**\n"
        
        return summary
    
    def get_latest_notifications(self, limit: int = 5) -> List[Dict]:
        """Get latest notifications."""
        return self.notifications[-limit:]


class SemanticChunker:
    """Semantic chunker for unstructured text."""
    
    def __init__(self, model, tokenizer, min_similarity: float = 0.70, stop_threshold: float = 0.75, max_sentences: int = 6):
        self.model = model
        self.tokenizer = tokenizer
        self.min_similarity = min_similarity
        self.stop_threshold = stop_threshold
        self.max_sentences = max_sentences
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        """Get embedding for a single sentence using mean pooling."""
        self.model.eval()
        with torch.no_grad():
            encoded = self.tokenizer(
                sentence,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            device = next(self.model.parameters()).device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            outputs = self.model.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
            embedding = torch.nn.functional.normalize(pooled, p=2, dim=1)
            
            return embedding.cpu().numpy()[0]
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text based on semantic similarity between sentences."""
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return sentences
        
        sentence_embeddings = [self.get_sentence_embedding(s) for s in sentences]
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embeddings = [sentence_embeddings[0]]
        
        for i in range(1, len(sentences)):
            chunk_mean_embedding = np.mean(current_embeddings, axis=0)
            chunk_mean_embedding = chunk_mean_embedding / norm(chunk_mean_embedding)
            
            next_embedding = sentence_embeddings[i]
            similarity = self.cosine_similarity(chunk_mean_embedding, next_embedding)
            
            should_stop = False
            
            if similarity < self.stop_threshold:
                should_stop = True
            elif len(current_chunk) >= self.max_sentences:
                should_stop = True
            elif similarity >= self.min_similarity:
                should_stop = False
            else:
                should_stop = True
            
            if should_stop:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_embeddings = [sentence_embeddings[i]]
            else:
                current_chunk.append(sentences[i])
                current_embeddings.append(sentence_embeddings[i])
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class VectorStore:
    """
    In-memory vector storage with similarity search.
    Compatible with PostgreSQL pgvector format.
    Stores: document_id, embedding_vector, data_source, metadata
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.vectors = {}  # document_id -> embedding
        self.metadata = {}  # document_id -> metadata dict
        self.data_sources = {}  # document_id -> 'postgresql' or 'unstructured'
        self.documents = {}  # document_id -> original content
        self.id_counter = 0
    
    def add(
        self,
        embedding: np.ndarray,
        content: Union[str, Dict],
        data_source: str = "unstructured",
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add vector to storage.
        
        Args:
            embedding: Embedding vector (768-dim)
            content: Original content (string or dict)
            data_source: 'postgresql' or 'unstructured'
            metadata: Optional metadata dict
        
        Returns:
            document_id
        """
        doc_id = self.id_counter
        self.id_counter += 1
        
        # Store embedding
        self.vectors[doc_id] = embedding
        self.data_sources[doc_id] = data_source
        self.documents[doc_id] = content
        
        # Store metadata
        if metadata is None:
            metadata = {}
        metadata['data_source'] = data_source
        metadata['embedding_dim'] = len(embedding)
        self.metadata[doc_id] = metadata
        
        return doc_id
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[int, float, Dict]]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of (doc_id, similarity_score, metadata) tuples
        """
        if not self.vectors:
            return []
        
        similarities = {}
        
        for doc_id, stored_embedding in self.vectors.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                norm(query_embedding) * norm(stored_embedding) + 1e-9
            )
            
            if similarity >= threshold:
                similarities[doc_id] = similarity
        
        # Sort by similarity (descending)
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Return top_k with metadata
        results = []
        for doc_id, similarity in sorted_results[:top_k]:
            results.append((doc_id, similarity, self.metadata[doc_id]))
        
        return results
    
    def get_vector(self, doc_id: int) -> Optional[np.ndarray]:
        """Get vector by document ID."""
        return self.vectors.get(doc_id)
    
    def get_document(self, doc_id: int) -> Optional[Union[str, Dict]]:
        """Get original document by document ID."""
        return self.documents.get(doc_id)
    
    def get_metadata(self, doc_id: int) -> Optional[Dict]:
        """Get metadata by document ID."""
        return self.metadata.get(doc_id)
    
    def delete(self, doc_id: int) -> bool:
        """Delete document from storage."""
        if doc_id in self.vectors:
            del self.vectors[doc_id]
            del self.metadata[doc_id]
            del self.data_sources[doc_id]
            del self.documents[doc_id]
            return True
        return False
    
    def size(self) -> int:
        """Get number of stored vectors."""
        return len(self.vectors)
    
    def export_for_pgvector(self, doc_id: int) -> Dict:
        """
        Export vector in PostgreSQL pgvector format.
        """
        if doc_id not in self.vectors:
            return None
        
        return {
            'id': doc_id,
            'embedding': self.vectors[doc_id].tolist(),
            'data_source': self.data_sources[doc_id],
            'metadata': self.metadata[doc_id],
            'content': str(self.documents[doc_id])
        }
    
    def batch_export_for_pgvector(self) -> List[Dict]:
        """Export all vectors in pgvector format."""
        return [self.export_for_pgvector(doc_id) for doc_id in self.vectors.keys()]


class DualPathRiskTransformer(nn.Module):
    """
    Dual-Path Risk Management Transformer with Vector Storage, Query Search & Notifications
    
    IF/ELSE ROUTING:
    IF data_source == "PostgreSQL" OR data is Dict/structured:
        → Use STRUCTURED_PATH
    ELSE (data is string/raw text):
        → Use UNSTRUCTURED_PATH
    
    FEATURES:
    - Stores all embeddings in VectorStore
    - Query search using cosine similarity
    - PostgreSQL pgvector compatible export
    - Notification system for risk alerts
    - Human-friendly outputs with bold formatting
    """
    
    def __init__(
        self,
        base_model_name: str = "sentence-transformers/distilbert-base-nli-mean-tokens",
        embedding_dim: int = 768,
        risk_categories: int = 3,
        severity_levels: int = 4,
        dropout_rate: float = 0.1,
        freeze_base: bool = False,
        chunk_min_similarity: float = 0.70,
        chunk_stop_threshold: float = 0.75,
        chunk_max_sentences: int = 6
    ):
        super(DualPathRiskTransformer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.risk_categories = risk_categories
        self.severity_levels = severity_levels
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(base_model_name)
        self.distilbert = DistilBertModel.from_pretrained(base_model_name)
        
        if freeze_base:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        self.pooling = models.Pooling(
            word_embedding_dimension=embedding_dim,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        
        # ===== STRUCTURED PATH (PostgreSQL) =====
        self.field_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.field_weight_network = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.structured_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # ===== UNSTRUCTURED PATH (Raw text) =====
        self.unstructured_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # ===== SHARED COMPONENTS =====
        self.category_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, risk_categories)
        )
        
        self.severity_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, severity_levels)
        )
        
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # ===== VECTOR STORAGE & NOTIFICATIONS =====
        self.vector_store = VectorStore(embedding_dim=embedding_dim)
        self.notification_system = NotificationSystem()
        
        # Semantic chunker for unstructured data
        self.semantic_chunker = SemanticChunker(
            self.distilbert,
            self.tokenizer,
            min_similarity=chunk_min_similarity,
            stop_threshold=chunk_stop_threshold,
            max_sentences=chunk_max_sentences
        )
        
        # Templates for structured data
        self.templates = {
            'default': "{key}: {value}",
            'risk_record': "Product ID: {product_id}, Risk Category: {risk_category}, Severity: {severity}, Region: {region}, Description: {description}",
            'incident': "Incident ID: {incident_id}, Type: {type}, Severity: {severity}, Date: {date}, Impact: {impact}, Resolution: {resolution}",
            'compliance': "Regulation: {regulation}, Status: {status}, Compliance Level: {compliance_level}, Last Audit: {last_audit}, Notes: {notes}",
            'quality': "Product: {product}, Defect Rate: {defect_rate}, Testing Phase: {phase}, Failed Tests: {failed_tests}, Root Cause: {root_cause}"
        }
        
        self.category_names = ["Product Risk", "Service Risk", "Brand Risk"]
        self.severity_names = ["Low", "Medium", "High", "Critical"]
    
    # ===== DATA FORMAT DETECTION =====
    
    def detect_data_source(self, data: Union[str, Dict]) -> tuple[str, str]:
        """
        Detect data source: PostgreSQL (structured) or raw text (unstructured).
        Returns: (source_type, format_name)
        """
        if isinstance(data, dict):
            return 'postgresql', "PostgreSQL Dictionary"
        
        if not isinstance(data, str):
            return 'postgresql', "Structured Object"
        
        text_stripped = data.strip()
        
        # Check for structured formats
        if text_stripped.startswith(('{', '[')):
            try:
                json.loads(text_stripped)
                return 'postgresql', "JSON"
            except:
                pass
        
        if re.search(r'INSERT\s+INTO\s+`?\w+`?\s*\(', text_stripped, re.IGNORECASE):
            return 'postgresql', "PostgreSQL Dump"
        
        lines = text_stripped.split('\n')
        kv_count = 0
        for line in lines[:10]:
            if re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_.-]*\s*[=:]\s*.+', line):
                kv_count += 1
        if kv_count > len(lines[:10]) * 0.5:
            return 'postgresql', "Key-Value Format"
        
        return 'unstructured', "Raw Text"
    
    def _format_with_template(self, data: Dict, template_name: str = 'default') -> str:
        """Format structured data using a template."""
        if template_name in self.templates:
            template = self.templates[template_name]
            try:
                return template.format(**data)
            except KeyError:
                pass
        
        # Default formatting
        formatted_parts = []
        for key, value in data.items():
            display_key = ' '.join(word.capitalize() for word in key.split('_'))
            
            if isinstance(value, float):
                if 0 < value < 1:
                    formatted_value = f"{value * 100:.1f}%"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, (list, tuple)):
                formatted_value = ", ".join(str(v) for v in value)
            else:
                formatted_value = str(value)
            
            formatted_parts.append(f"{display_key}: {formatted_value}")
        
        return " | ".join(formatted_parts)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        data_source: str = "unstructured",
        return_classifications: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with IF/ELSE routing."""
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_embeddings = sum_embeddings / sum_mask
        
        # ===== IF/ELSE ROUTING =====
        if data_source == "postgresql":
            # STRUCTURED PATH
            field_enhanced, _ = self.field_attention(
                pooled_embeddings.unsqueeze(1),
                pooled_embeddings.unsqueeze(1),
                pooled_embeddings.unsqueeze(1)
            )
            field_enhanced = field_enhanced.squeeze(1)
            processed_embeddings = self.structured_encoder(field_enhanced)
        else:
            # UNSTRUCTURED PATH
            processed_embeddings = self.unstructured_encoder(pooled_embeddings)
        
        # Project and normalize
        final_embeddings = self.projection(processed_embeddings)
        final_embeddings = self.layer_norm(final_embeddings)
        final_embeddings = torch.nn.functional.normalize(final_embeddings, p=2, dim=1)
        
        result = {
            'embeddings': final_embeddings,
            'pooled_output': pooled_embeddings,
            'data_source': data_source
        }
        
        if return_classifications:
            category_logits = self.category_classifier(processed_embeddings)
            severity_logits = self.severity_classifier(processed_embeddings)
            
            result['category_logits'] = category_logits
            result['severity_logits'] = severity_logits
            result['category_probs'] = torch.softmax(category_logits, dim=-1)
            result['severity_probs'] = torch.softmax(severity_logits, dim=-1)
        
        return result
    
    def encode_data(
        self,
        data: Union[str, Dict, List[Union[str, Dict]]],
        template_name: str = 'default',
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        show_formatting: bool = False,
        store_vectors: bool = True,
        metadata: Optional[Dict] = None,
        check_alerts: bool = True
    ):
        """
        Encode data and optionally store vectors with notifications.
        
        Args:
            data: Input data
            template_name: Template for structured data
            batch_size: Batch size
            convert_to_numpy: Return numpy arrays
            show_formatting: Print formatting
            store_vectors: Store in vector store
            metadata: Optional metadata
            check_alerts: Check for risk alerts
        
        Returns:
            embeddings, sources, document_ids, alerts (if checking)
        """
        if isinstance(data, (dict, str)):
            data = [data]
        
        self.eval()
        embeddings = []
        sources = []
        document_ids = []
        alerts = []
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                batch_sources = []
                batch_texts = []
                
                # Prepare each item
                for item in batch:
                    # ===== SCHEMA ON WRITE VALIDATION =====
                    data_source, format_name = self.detect_data_source(item)
                    
                    # Validate input data
                    is_valid, validation_errors = SchemaValidator.validate_schema_on_write(
                        item, 'structured' if data_source == 'postgresql' else 'unstructured'
                    )
                    
                    if not is_valid:
                        print(f"[VALIDATION ERROR] {item}")
                        for error in validation_errors:
                            print(f"   Error: {error}")
                        continue  # Skip invalid data
                    
                    batch_sources.append((data_source, format_name))
                    
                    if data_source == "postgresql":
                        if isinstance(item, dict):
                            prepared_text = self._format_with_template(item, template_name)
                        else:
                            prepared_text = str(item)
                    else:
                        prepared_text = str(item)
                    
                    batch_texts.append(prepared_text)
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                device = next(self.parameters()).device
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                # Process each item
                for j, (data_source, format_name) in enumerate(batch_sources):
                    single_input_ids = input_ids[j:j+1]
                    single_attention_mask = attention_mask[j:j+1]
                    
                    outputs = self.forward(
                        single_input_ids,
                        single_attention_mask,
                        data_source=data_source,
                        return_classifications=True
                    )
                    
                    embedding = outputs['embeddings'].cpu().numpy()[0]
                    
                    if convert_to_numpy:
                        embeddings.append(embedding)
                    else:
                        embeddings.append(outputs['embeddings'])
                    
                    # Get predictions for alerts
                    category_idx = outputs['category_probs'][0].argmax()
                    severity_idx = outputs['severity_probs'][0].argmax()
                    category_confidence = float(outputs['category_probs'][0][category_idx])
                    severity_confidence = float(outputs['severity_probs'][0][severity_idx])
                    
                    # Store vector if requested
                    if store_vectors:
                        meta = metadata.copy() if metadata else {}
                        meta['format'] = format_name
                        meta['category'] = self.category_names[category_idx]
                        meta['severity'] = self.severity_names[severity_idx]
                        meta['category_confidence'] = category_confidence
                        meta['severity_confidence'] = severity_confidence
                        
                        doc_id = self.vector_store.add(
                            embedding=embedding,
                            content=batch[j],
                            data_source=data_source,
                            metadata=meta
                        )
                        document_ids.append(doc_id)
                        
                        # Check for alerts
                        if check_alerts and self.notification_system.check_threshold_breach(
                            severity_confidence, 'high'
                        ):
                            alert = self.notification_system.create_notification(
                                doc_id=doc_id,
                                title=f"{self.category_names[category_idx]} Detected",
                                description=str(batch[j])[:150],
                                confidence=severity_confidence,
                                risk_category=self.category_names[category_idx],
                                severity_level=self.severity_names[severity_idx],
                                data_source=data_source
                            )
                            
                            # ===== SCHEMA ON READ VALIDATION =====
                            is_alert_valid, alert_errors = SchemaValidator.validate_schema_on_read(alert, 'alert')
                            if is_alert_valid:
                                alerts.append(alert)
                            else:
                                print(f"[ALERT VALIDATION ERROR] Doc #{doc_id}")
                                for error in alert_errors:
                                    print(f"   Error: {error}")
                    
                    sources.append((data_source, format_name))
        
        if convert_to_numpy:
            embeddings = np.vstack(embeddings)
        
        if store_vectors:
            return embeddings, sources, document_ids, alerts
        else:
            return embeddings, sources
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query for semantic search."""
        self.eval()
        
        with torch.no_grad():
            encoded = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            device = next(self.parameters()).device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            outputs = self.forward(
                input_ids,
                attention_mask,
                data_source="unstructured"
            )
            
            return outputs['embeddings'].cpu().numpy()[0]
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> str:
        """
        Semantic search in stored vectors with human-friendly output.
        
        Args:
            query: Query text
            top_k: Number of results
            threshold: Minimum similarity threshold
        
        Returns:
            Human-friendly formatted search results
        """
        query_embedding = self.encode_query(query)
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            threshold=threshold
        )
        
        if not results:
            return "No matching risks found in database."
        
        output = f"\nSEMANTIC SEARCH RESULTS (Query: {query})\n\n"
        output += f"Found **{len(results)}** matching results:\n\n"
        
        for idx, (doc_id, similarity, metadata) in enumerate(results, 1):
            document = self.vector_store.get_document(doc_id)
            similarity_pct = int(similarity * 100)
            
            output += f"**{idx}. Result #{doc_id}** - Similarity: **{similarity_pct}%**\n"
            output += f"   Source: {metadata['data_source']}\n"
            output += f"   Category: **{metadata.get('category', 'Unknown')}**\n"
            output += f"   Severity: **{metadata.get('severity', 'Unknown')}**\n"
            output += f"   Content: {str(document)[:100]}...\n\n"
        
        return output
    
    def predict_risk_attributes(
        self,
        data: Union[str, Dict, List[Union[str, Dict]]],
        template_name: str = 'default',
        batch_size: int = 32
    ) -> str:
        """Predict risk with human-friendly output."""
        if isinstance(data, (dict, str)):
            data = [data]
        
        self.eval()
        all_categories = []
        all_severities = []
        sources = []
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                batch_sources = []
                batch_texts = []
                
                for item in batch:
                    data_source, format_name = self.detect_data_source(item)
                    batch_sources.append(data_source)
                    
                    if data_source == "postgresql" and isinstance(item, dict):
                        prepared_text = self._format_with_template(item, template_name)
                    else:
                        prepared_text = str(item)
                    
                    batch_texts.append(prepared_text)
                
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                device = next(self.parameters()).device
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                for j, data_source in enumerate(batch_sources):
                    single_input_ids = input_ids[j:j+1]
                    single_attention_mask = attention_mask[j:j+1]
                    
                    outputs = self.forward(
                        single_input_ids,
                        single_attention_mask,
                        data_source=data_source,
                        return_classifications=True
                    )
                    
                    all_categories.append(outputs['category_probs'].cpu().numpy())
                    all_severities.append(outputs['severity_probs'].cpu().numpy())
                
                sources.extend(batch_sources)
        
        # Format results
        categories = np.vstack(all_categories).argmax(axis=1)
        severities = np.vstack(all_severities).argmax(axis=1)
        cat_probs = np.vstack(all_categories)
        sev_probs = np.vstack(all_severities)
        
        output = "\nRISK ANALYSIS RESULTS\n\n"
        
        for idx, (cat_idx, sev_idx) in enumerate(zip(categories, severities)):
            cat_name = self.category_names[cat_idx]
            sev_name = self.severity_names[sev_idx]
            cat_conf = int(cat_probs[idx][cat_idx] * 100)
            sev_conf = int(sev_probs[idx][sev_idx] * 100)
            
            output += f"**Analysis {idx + 1}:**\n"
            output += f"   Risk Category: **{cat_name}** (**{cat_conf}%** confidence)\n"
            output += f"   Severity Level: **{sev_name}** (**{sev_conf}%** confidence)\n"
            output += f"   Data Source: **{sources[idx]}**\n\n"
        
        return output
    
    def get_alert_summary(self) -> str:
        """Get human-friendly alert summary."""
        return self.notification_system.get_notifications_summary()
    
    def get_latest_alerts(self, limit: int = 5) -> List[Dict]:
        """Get latest alerts."""
        return self.notification_system.get_latest_notifications(limit)
    
    def get_vector_store_stats(self) -> str:
        """Get vector store statistics as human-friendly text."""
        size = self.vector_store.size()
        return f"\nVECTOR STORE STATISTICS\n\nTotal Vectors Stored: **{size}**\nEmbedding Dimension: **{self.embedding_dim}**"
    
    def export_to_pgvector(self) -> List[Dict]:
        """Export all vectors in PostgreSQL pgvector format."""
        return self.vector_store.batch_export_for_pgvector()
    
    def save_model(self, path: str):
        """Save model weights and configuration."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'embedding_dim': self.embedding_dim,
                'risk_categories': self.risk_categories,
                'severity_levels': self.severity_levels
            },
            'templates': self.templates
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'templates' in checkpoint:
            self.templates = checkpoint['templates']
        print(f"Model loaded from {path}")


def create_dual_path_transformer(
    base_model: str = "sentence-transformers/distilbert-base-nli-mean-tokens",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> DualPathRiskTransformer:
    """Factory function to create the Dual-Path Risk Transformer."""
    model = DualPathRiskTransformer(base_model_name=base_model)
    model = model.to(device)
    return model
