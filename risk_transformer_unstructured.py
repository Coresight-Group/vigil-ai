"""
Modified DistilBERT-based Sentence Transformer for Risk Management
Extends sentence-transformers/distilbert-base-nli-mean-tokens with:
- Risk-specific fine-tuning layers
- Semantic chunking with cosine similarity thresholds
- Structured data encoding
- Multi-task learning for risk categorization
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
from sentence_transformers import SentenceTransformer, models
import json
import re
from typing import Dict, List, Union, Optional, Tuple
import numpy as np
from numpy.linalg import norm


class SemanticChunker:
    """
    Semantic chunker that splits text based on sentence similarity.
    Chunks continue if cosine similarity >= 0.70 and max 6 sentences.
    Chunks stop if next sentence similarity < 0.75.
    """
    
    def __init__(self, model, tokenizer, min_similarity: float = 0.70, stop_threshold: float = 0.75, max_sentences: int = 6):
        self.model = model
        self.tokenizer = tokenizer
        self.min_similarity = min_similarity
        self.stop_threshold = stop_threshold
        self.max_sentences = max_sentences
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Enhanced sentence splitting pattern
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        """Get embedding for a single sentence using mean pooling."""
        self.model.eval()
        with torch.no_grad():
            # Tokenize
            encoded = self.tokenizer(
                sentence,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Get embeddings from DistilBERT
            outputs = self.model.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
            # Normalize for cosine similarity
            embedding = torch.nn.functional.normalize(pooled, p=2, dim=1)
            
            return embedding.cpu().numpy()[0]
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text based on semantic similarity between sentences.
        
        Rules:
        - Continue adding sentences if similarity >= 0.70 AND chunk has < 6 sentences
        - STOP if next sentence similarity < 0.75
        - Maximum 6 sentences per chunk
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return sentences
        
        # Get embeddings for all sentences
        print(f"Computing embeddings for {len(sentences)} sentences...")
        sentence_embeddings = [self.get_sentence_embedding(s) for s in sentences]
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embeddings = [sentence_embeddings[0]]
        
        for i in range(1, len(sentences)):
            # Calculate mean embedding of current chunk
            chunk_mean_embedding = np.mean(current_embeddings, axis=0)
            chunk_mean_embedding = chunk_mean_embedding / norm(chunk_mean_embedding)
            
            # Calculate similarity with next sentence
            next_embedding = sentence_embeddings[i]
            similarity = self.cosine_similarity(chunk_mean_embedding, next_embedding)
            
            print(f"Sentence {i}: Similarity = {similarity:.3f} | Chunk size = {len(current_chunk)}")
            
            # Decision logic
            should_stop = False
            
            # RULE 1: Stop if similarity < 0.75 (hard stop threshold)
            if similarity < self.stop_threshold:
                should_stop = True
                print(f"  → STOP: Similarity {similarity:.3f} < {self.stop_threshold}")
            
            # RULE 2: Stop if already at max sentences
            elif len(current_chunk) >= self.max_sentences:
                should_stop = True
                print(f"  → STOP: Max sentences ({self.max_sentences}) reached")
            
            # RULE 3: Continue if similarity >= 0.70 and not at max
            elif similarity >= self.min_similarity:
                should_stop = False
                print(f"  → CONTINUE: Similarity {similarity:.3f} >= {self.min_similarity}")
            
            # RULE 4: Stop if similarity between 0.70 and 0.75
            else:
                should_stop = True
                print(f"  → STOP: Similarity {similarity:.3f} below minimum {self.min_similarity}")
            
            if should_stop:
                # Save current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_embeddings = [sentence_embeddings[i]]
            else:
                # Add sentence to current chunk
                current_chunk.append(sentences[i])
                current_embeddings.append(sentence_embeddings[i])
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class RiskManagementTransformer(nn.Module):
    """
    Custom transformer model for risk management embedding generation.
    Based on distilbert-base-nli-mean-tokens with risk-specific modifications.
    Includes semantic chunking capability.
    """
    
    def __init__(
        self,
        base_model_name: str = "sentence-transformers/distilbert-base-nli-mean-tokens",
        embedding_dim: int = 768,
        risk_categories: int = 3,  # product, service, brand
        severity_levels: int = 4,  # low, medium, high, critical
        dropout_rate: float = 0.1,
        freeze_base: bool = False,
        chunk_min_similarity: float = 0.70,
        chunk_stop_threshold: float = 0.75,
        chunk_max_sentences: int = 6
    ):
        super(RiskManagementTransformer, self).__init__()
        
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
        
        # Mean pooling layer (matching sentence-transformers approach)
        self.pooling = models.Pooling(
            word_embedding_dimension=embedding_dim,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        
        # Risk-specific enhancement layers
        self.risk_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Structured data encoder (for JSON/tabular data)
        self.structured_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Risk category classification head (multi-task learning)
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
        
        # Final projection layer for normalized embeddings
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Initialize semantic chunker
        self.chunker = SemanticChunker(
            model=self,
            tokenizer=self.tokenizer,
            min_similarity=chunk_min_similarity,
            stop_threshold=chunk_stop_threshold,
            max_sentences=chunk_max_sentences
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        structured_features: Optional[torch.Tensor] = None,
        return_classifications: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized text input
            attention_mask: Attention mask for padding
            structured_features: Optional pre-encoded structured data features
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
        
        # Apply risk-specific attention
        risk_enhanced, _ = self.risk_attention(
            pooled_embeddings.unsqueeze(1),
            pooled_embeddings.unsqueeze(1),
            pooled_embeddings.unsqueeze(1)
        )
        risk_enhanced = risk_enhanced.squeeze(1)
        
        # Combine with structured data if provided
        if structured_features is not None:
            structured_encoded = self.structured_encoder(structured_features)
            combined_embeddings = risk_enhanced + structured_encoded
        else:
            combined_embeddings = risk_enhanced
        
        # Project and normalize
        final_embeddings = self.projection(combined_embeddings)
        final_embeddings = self.layer_norm(final_embeddings)
        
        # Normalize embeddings for cosine similarity
        final_embeddings = torch.nn.functional.normalize(final_embeddings, p=2, dim=1)
        
        result = {
            'embeddings': final_embeddings,
            'pooled_output': pooled_embeddings
        }
        
        # Optional classification outputs for multi-task learning
        if return_classifications:
            category_logits = self.category_classifier(combined_embeddings)
            severity_logits = self.severity_classifier(combined_embeddings)
            
            result['category_logits'] = category_logits
            result['severity_logits'] = severity_logits
            result['category_probs'] = torch.softmax(category_logits, dim=-1)
            result['severity_probs'] = torch.softmax(severity_logits, dim=-1)
        
        return result
    
    def chunk_and_embed(
        self,
        text: str,
        convert_to_numpy: bool = True
    ) -> Tuple[List[str], Union[torch.Tensor, np.ndarray]]:
        """
        Chunk text semantically and generate embeddings for each chunk.
        
        Args:
            text: Input text to chunk and embed
            convert_to_numpy: Convert output to numpy array
            
        Returns:
            Tuple of (list of chunks, embeddings for each chunk)
        """
        # Perform semantic chunking
        chunks = self.chunker.chunk_text(text)
        
        print(f"\nCreated {len(chunks)} semantic chunks")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {len(chunk.split())} words")
        
        # Generate embeddings for all chunks
        embeddings = self.encode_text(chunks, convert_to_numpy=convert_to_numpy)
        
        return chunks, embeddings
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode text inputs into embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
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
    
    def encode_structured_data(
        self,
        structured_data: Union[Dict, List[Dict]],
        template: str = "Risk Category: {category}, Severity: {severity}, Product: {product}, Description: {description}",
        batch_size: int = 32,
        convert_to_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode structured risk data (JSON/dict) into embeddings.
        
        Args:
            structured_data: Dictionary or list of dictionaries with risk data
            template: Text template for formatting structured data
            batch_size: Batch size for encoding
            convert_to_numpy: Convert output to numpy array
            
        Returns:
            Structured data embeddings
        """
        if isinstance(structured_data, dict):
            structured_data = [structured_data]
        
        # Convert structured data to text using template
        texts = []
        for data in structured_data:
            try:
                # Fill template with available fields
                text = template.format(**data)
            except KeyError:
                # Fallback: JSON string representation
                text = json.dumps(data, indent=None)
            texts.append(text)
        
        # Use text encoding with structured data flag
        return self.encode_text(
            texts,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy
        )
    
    def predict_risk_attributes(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Predict risk category and severity level for input texts.
        
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
    
    def save_model(self, path: str):
        """Save model weights and configuration."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'embedding_dim': self.embedding_dim,
                'risk_categories': self.risk_categories,
                'severity_levels': self.severity_levels
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


class RiskDataset(torch.utils.data.Dataset):
    """
    Dataset class for training the Risk Management Transformer.
    Handles both text and structured data inputs.
    """
    
    def __init__(
        self,
        texts: List[str],
        categories: Optional[List[int]] = None,
        severities: Optional[List[int]] = None,
        structured_data: Optional[List[Dict]] = None,
        tokenizer: DistilBertTokenizer = None,
        max_length: int = 512
    ):
        self.texts = texts
        self.categories = categories
        self.severities = severities
        self.structured_data = structured_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        
        # Add labels if available
        if self.categories is not None:
            item['category'] = torch.tensor(self.categories[idx], dtype=torch.long)
        
        if self.severities is not None:
            item['severity'] = torch.tensor(self.severities[idx], dtype=torch.long)
        
        return item


def create_risk_transformer(
    base_model: str = "sentence-transformers/distilbert-base-nli-mean-tokens",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> RiskManagementTransformer:
    """
    Factory function to create and initialize the Risk Management Transformer.
    
    Args:
        base_model: Base model name or path
        device: Device to load model on
        
    Returns:
        Initialized RiskManagementTransformer
    """
    model = RiskManagementTransformer(base_model_name=base_model)
    model = model.to(device)
    return model


# Example usage and training loop
if __name__ == "__main__":
    # Initialize model
    model = create_risk_transformer()
    
    # Example: Semantic chunking with embedding
    long_text = """
    Our supply chain faces critical disruption risks in the Asian manufacturing region. 
    The delays have cascaded through our logistics network. 
    Quality control inspections reveal increased defect rates in components. 
    Testing protocols show failure rates above acceptable thresholds. 
    Brand reputation monitoring indicates negative sentiment on social media. 
    Customer complaints about delayed shipments are increasing. 
    Service level agreements are at risk of breach if issues persist.
    """
    
    print("=== SEMANTIC CHUNKING DEMO ===")
    chunks, chunk_embeddings = model.chunk_and_embed(long_text)
    
    print("\n=== CHUNKS CREATED ===")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Text: {chunk[:100]}...")
        print(f"  Embedding shape: {chunk_embeddings[i].shape}")
    
    # Example: Encode text
    sample_texts = [
        "Critical supply chain disruption affecting Product X manufacturing",
        "Service level agreement breach - 99.9% uptime not met for 3 consecutive days",
        "Brand reputation risk - negative social media sentiment trending"
    ]
    
    embeddings = model.encode_text(sample_texts)
    print(f"\nText embeddings shape: {embeddings.shape}")
    
    # Example: Encode structured data
    sample_structured = [
        {
            "category": "product",
            "severity": "critical",
            "product": "Product X",
            "description": "Supply chain disruption"
        },
        {
            "category": "service",
            "severity": "high",
            "product": "Service API",
            "description": "SLA breach"
        }
    ]
    
    structured_embeddings = model.encode_structured_data(sample_structured)
    print(f"Structured data embeddings shape: {structured_embeddings.shape}")
    
    # Example: Predict risk attributes
    predictions = model.predict_risk_attributes(sample_texts)
    print(f"Predicted categories: {predictions['predicted_categories']}")
    print(f"Predicted severities: {predictions['predicted_severities']}")
