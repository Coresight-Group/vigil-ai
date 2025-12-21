"""
DistilBERT-based Transformer for Structured Data Embedding
Optimized specifically for tabular/structured data (database rows, JSON objects)
NO semantic chunking - each row is treated as a complete semantic unit
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sentence_transformers import models
import json
from typing import Dict, List, Union, Optional
import numpy as np


class StructuredDataTransformer(nn.Module):
    """
    Transformer model optimized for structured data embedding.
    Designed for database rows, JSON objects, and tabular data.
    Each structured record is embedded as a single semantic unit.
    """
    
    def __init__(
        self,
        base_model_name: str = "sentence-transformers/distilbert-base-nli-mean-tokens",
        embedding_dim: int = 768,
        risk_categories: int = 3,  # product, service, brand
        severity_levels: int = 4,  # low, medium, high, critical
        dropout_rate: float = 0.1,
        freeze_base: bool = False
    ):
        super(StructuredDataTransformer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.risk_categories = risk_categories
        self.severity_levels = severity_levels
        
        # Load base DistilBERT model
        self.tokenizer = DistilBertTokenizer.from_pretrained(base_model_name)
        self.distilbert = DistilBertModel.from_pretrained(base_model_name)
        
        # Optionally freeze base model for transfer learning
        if freeze_base:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        # Mean pooling layer
        self.pooling = models.Pooling(
            word_embedding_dimension=embedding_dim,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        
        # Structured data enhancement layers
        self.field_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Field importance weighting (learns which fields matter most)
        self.field_weight_network = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Structured data-specific encoder
        self.structured_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Risk category classification head
        self.category_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, risk_categories)
        )
        
        # Severity level classification head
        self.severity_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, severity_levels)
        )
        
        # Final projection layer
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Template configurations for different data types
        self.templates = {
            'default': "{key}: {value}",
            'risk_record': "Product ID: {product_id}, Risk Category: {risk_category}, Severity: {severity}, Region: {region}, Description: {description}",
            'incident': "Incident ID: {incident_id}, Type: {type}, Severity: {severity}, Date: {date}, Impact: {impact}, Resolution: {resolution}",
            'compliance': "Regulation: {regulation}, Status: {status}, Compliance Level: {compliance_level}, Last Audit: {last_audit}, Notes: {notes}",
            'quality': "Product: {product}, Defect Rate: {defect_rate}, Testing Phase: {phase}, Failed Tests: {failed_tests}, Root Cause: {root_cause}"
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_classifications: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized text input
            attention_mask: Attention mask for padding
            return_classifications: Whether to return risk category and severity predictions
            
        Returns:
            Dictionary containing embeddings and optional classifications
        """
        # Get DistilBERT embeddings
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Token embeddings from last hidden state
        token_embeddings = outputs.last_hidden_state
        
        # Mean pooling (weighted by attention mask)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_embeddings = sum_embeddings / sum_mask
        
        # Apply field-specific attention
        field_enhanced, _ = self.field_attention(
            pooled_embeddings.unsqueeze(1),
            pooled_embeddings.unsqueeze(1),
            pooled_embeddings.unsqueeze(1)
        )
        field_enhanced = field_enhanced.squeeze(1)
        
        # Apply structured data encoder
        structured_embeddings = self.structured_encoder(field_enhanced)
        
        # Project and normalize
        final_embeddings = self.projection(structured_embeddings)
        final_embeddings = self.layer_norm(final_embeddings)
        
        # Normalize embeddings for cosine similarity
        final_embeddings = torch.nn.functional.normalize(final_embeddings, p=2, dim=1)
        
        result = {
            'embeddings': final_embeddings,
            'pooled_output': pooled_embeddings
        }
        
        # Optional classification outputs
        if return_classifications:
            category_logits = self.category_classifier(structured_embeddings)
            severity_logits = self.severity_classifier(structured_embeddings)
            
            result['category_logits'] = category_logits
            result['severity_logits'] = severity_logits
            result['category_probs'] = torch.softmax(category_logits, dim=-1)
            result['severity_probs'] = torch.softmax(severity_logits, dim=-1)
        
        return result
    
    def _format_with_template(
        self,
        data: Dict,
        template_name: str = 'default'
    ) -> str:
        """
        Format structured data using a template.
        
        Args:
            data: Dictionary with structured data
            template_name: Name of template to use
            
        Returns:
            Formatted text string
        """
        # Get template
        if template_name in self.templates:
            template = self.templates[template_name]
            try:
                return template.format(**data)
            except KeyError:
                # Fallback to default if template fields don't match
                pass
        
        # Default formatting: "key: value" for all fields
        formatted_parts = []
        for key, value in data.items():
            # Convert key from snake_case to Title Case
            display_key = ' '.join(word.capitalize() for word in key.split('_'))
            
            # Format value appropriately
            if isinstance(value, float):
                if 0 < value < 1:
                    # Likely a percentage or rate
                    formatted_value = f"{value * 100:.1f}%"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, (list, tuple)):
                formatted_value = ", ".join(str(v) for v in value)
            else:
                formatted_value = str(value)
            
            formatted_parts.append(f"{display_key}: {formatted_value}")
        
        return " | ".join(formatted_parts)
    
    def encode_structured_data(
        self,
        structured_data: Union[Dict, List[Dict]],
        template_name: str = 'default',
        custom_template: Optional[str] = None,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        show_formatting: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode structured data (JSON/dict/table rows) into embeddings.
        
        Args:
            structured_data: Dictionary or list of dictionaries
            template_name: Name of predefined template ('default', 'risk_record', 'incident', 'compliance', 'quality')
            custom_template: Custom template string (overrides template_name)
            batch_size: Batch size for encoding
            convert_to_numpy: Convert output to numpy array
            show_formatting: Print formatted text for debugging
            
        Returns:
            Embeddings for structured data
        """
        if isinstance(structured_data, dict):
            structured_data = [structured_data]
        
        # Convert structured data to text
        texts = []
        for data in structured_data:
            if custom_template:
                try:
                    text = custom_template.format(**data)
                except KeyError as e:
                    print(f"Warning: Template key {e} not found in data, using default formatting")
                    text = self._format_with_template(data, 'default')
            else:
                text = self._format_with_template(data, template_name)
            
            texts.append(text)
            
            if show_formatting:
                print(f"Formatted: {text}")
        
        # Encode the formatted text
        return self.encode_text(
            texts,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy
        )
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode text inputs into embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            convert_to_numpy: Convert output to numpy array
            normalize_embeddings: L2 normalize embeddings
            
        Returns:
            Text embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        self.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(next(self.parameters()).device)
                attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
                
                # Generate embeddings
                outputs = self.forward(input_ids, attention_mask)
                embeddings = outputs['embeddings']
                
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return all_embeddings.numpy()
        return all_embeddings
    
    def batch_encode_database_table(
        self,
        rows: List[Dict],
        template_name: str = 'default',
        batch_size: int = 64,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Efficiently encode an entire database table.
        Optimized for large-scale structured data processing.
        
        Args:
            rows: List of dictionaries (database rows)
            template_name: Template to use for formatting
            batch_size: Batch size for processing
            show_progress: Show progress
            
        Returns:
            Numpy array of embeddings (one per row)
        """
        total_rows = len(rows)
        
        if show_progress:
            print(f"Encoding {total_rows} database rows...")
        
        embeddings = self.encode_structured_data(
            rows,
            template_name=template_name,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        
        if show_progress:
            print(f"✓ Successfully encoded {total_rows} rows")
            print(f"  Embedding shape: {embeddings.shape}")
        
        return embeddings
    
    def predict_risk_attributes(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Predict risk category and severity level.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary with category and severity predictions
        """
        if isinstance(texts, str):
            texts = [texts]
        
        self.eval()
        all_categories = []
        all_severities = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(next(self.parameters()).device)
                attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
                
                # Get predictions
                outputs = self.forward(
                    input_ids,
                    attention_mask,
                    return_classifications=True
                )
                
                all_categories.append(outputs['category_probs'].cpu().numpy())
                all_severities.append(outputs['severity_probs'].cpu().numpy())
        
        return {
            'category_probabilities': np.vstack(all_categories),
            'severity_probabilities': np.vstack(all_severities),
            'predicted_categories': np.vstack(all_categories).argmax(axis=1),
            'predicted_severities': np.vstack(all_severities).argmax(axis=1)
        }
    
    def add_custom_template(self, name: str, template: str):
        """
        Add a custom template for formatting structured data.
        
        Args:
            name: Template name
            template: Template string with {field_name} placeholders
            
        Example:
            model.add_custom_template(
                'my_template',
                'Item: {item_id}, Status: {status}, Value: {value}'
            )
        """
        self.templates[name] = template
        print(f"Added custom template '{name}'")
    
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


def create_structured_transformer(
    base_model: str = "sentence-transformers/distilbert-base-nli-mean-tokens",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> StructuredDataTransformer:
    """
    Factory function to create the Structured Data Transformer.
    
    Args:
        base_model: Base model name or path
        device: Device to load model on
        
    Returns:
        Initialized StructuredDataTransformer
    """
    model = StructuredDataTransformer(base_model_name=base_model)
    model = model.to(device)
    return model


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = create_structured_transformer()
    
    print("="*60)
    print("STRUCTURED DATA TRANSFORMER - Example Usage")
    print("="*60)
    
    # Example 1: Single structured record
    print("\n1. SINGLE RECORD ENCODING")
    print("-" * 60)
    
    single_record = {
        "product_id": "PROD_001",
        "risk_category": "Supply Chain",
        "severity": "Critical",
        "defect_rate": 0.15,
        "region": "Asia",
        "description": "Manufacturing delays due to supplier issues"
    }
    
    embedding = model.encode_structured_data(
        single_record,
        template_name='risk_record',
        show_formatting=True
    )
    print(f"Embedding shape: {embedding.shape}")
    
    # Example 2: Multiple records (database table)
    print("\n\n2. BATCH DATABASE TABLE ENCODING")
    print("-" * 60)
    
    database_rows = [
        {
            "product_id": "PROD_001",
            "risk_category": "Supply Chain",
            "severity": "Critical",
            "defect_rate": 0.15,
            "region": "Asia",
            "description": "Manufacturing delays"
        },
        {
            "product_id": "PROD_002",
            "risk_category": "Quality Control",
            "severity": "High",
            "defect_rate": 0.08,
            "region": "Europe",
            "description": "Component failures"
        },
        {
            "product_id": "PROD_003",
            "risk_category": "Brand Risk",
            "severity": "Medium",
            "defect_rate": 0.02,
            "region": "North America",
            "description": "Negative sentiment"
        }
    ]
    
    embeddings = model.batch_encode_database_table(
        database_rows,
        template_name='risk_record',
        batch_size=32
    )
    
    # Example 3: Custom template
    print("\n\n3. CUSTOM TEMPLATE")
    print("-" * 60)
    
    model.add_custom_template(
        'custom_risk',
        'ALERT: {severity} risk in {region} - Product {product_id} has {defect_rate} defect rate. Issue: {description}'
    )
    
    custom_embedding = model.encode_structured_data(
        single_record,
        template_name='custom_risk',
        show_formatting=True
    )
    
    # Example 4: Different data types
    print("\n\n4. DIFFERENT STRUCTURED DATA TYPES")
    print("-" * 60)
    
    incident_record = {
        "incident_id": "INC_2024_001",
        "type": "Security Breach",
        "severity": "Critical",
        "date": "2024-01-15",
        "impact": "Customer data exposed",
        "resolution": "Patched vulnerability, notified customers"
    }
    
    incident_embedding = model.encode_structured_data(
        incident_record,
        template_name='incident',
        show_formatting=True
    )
    
    # Example 5: Query search
    print("\n\n5. SEMANTIC SEARCH EXAMPLE")
    print("-" * 60)
    
    query = "Critical supply chain problems in Asia"
    query_embedding = model.encode_text(query)
    
    # Calculate cosine similarity with database records
    from numpy.linalg import norm
    
    print(f"\nQuery: '{query}'")
    print("\nSimilarity scores:")
    for i, row in enumerate(database_rows):
        similarity = np.dot(query_embedding[0], embeddings[i]) / (
            norm(query_embedding[0]) * norm(embeddings[i])
        )
        print(f"  Row {i+1} ({row['product_id']}): {similarity:.4f}")
    
    print("\n" + "="*60)
    print("✓ All examples completed successfully!")
    print("="*60)
