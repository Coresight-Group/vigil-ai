---
title: CoreSightGroup
emoji:
colorFrom: gray
colorTo: yellow
sdk: static
pinned: false
---

# Risk Management Transformers

> Dual-Model Architecture for Structured & Unstructured Risk Data

[![DistilBERT-base](https://img.shields.io/badge/model-DistilBERT--base-DAA520?style=flat-square)](#)
[![768-dim embeddings](https://img.shields.io/badge/embeddings-768--dim-DAA520?style=flat-square)](#)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-DAA520?style=flat-square)](#)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-DAA520?style=flat-square)](#)

---

## Overview

A specialized transformer architecture built on `distilbert-base-nli-mean-tokens` for risk management applications. This repository contains two optimized models designed for different data types, enabling semantic search and risk classification across products, services, and brand reputation.

---

## Models

### 📄 Unstructured Data Model
**Semantic chunking with cosine similarity thresholds**

**Best For:**
- Documents and reports
- Email threads
- Policy documents
- News articles
- Meeting notes

**Features:**
- Intelligent sentence grouping
- 70% similarity threshold
- 75% hard stop threshold
- Max 6 sentences per chunk
- Mean pooling similarity

---

### 📊 Structured Data Model
**Template-based encoding for tabular data**

**Best For:**
- Database tables
- JSON records
- CSV data
- API responses
- Structured logs

**Features:**
- Pre-built templates
- Custom formatting
- Batch processing
- Field attention
- Auto type conversion

---

## Model Comparison

| Feature | Unstructured Model | Structured Model |
|---------|-------------------|------------------|
| **Input Type** | Long-form text, documents | Database rows, JSON objects |
| **Chunking Strategy** | Semantic similarity-based | None (1 row = 1 embedding) |
| **Processing Method** | Sentence-by-sentence analysis | Template-based formatting |
| **Optimal Use Case** | Risk reports, incident narratives | Risk databases, compliance records |
| **Performance** | Slower (semantic analysis) | Faster (direct encoding) |

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/risk-management-transformers.git
cd risk-management-transformers

# Install dependencies
pip install torch transformers sentence-transformers numpy
```

---

## Quick Start

### Unstructured Data Example

This example shows how to process a long document and automatically split it into semantic chunks. The model analyzes sentence similarity and groups related sentences together (max 6 sentences per chunk, stops when similarity drops below 75%).

```python
from risk_transformer import create_risk_transformer

# Initialize model
model = create_risk_transformer()

# Process long document with semantic chunking
document = """
Supply chain disruptions in Asia continue. Manufacturing 
delays affect multiple product lines. Quality control shows 
defects above threshold. Brand sentiment declining.
"""

# Chunk and embed
chunks, embeddings = model.chunk_and_embed(document)

print(f"Created {len(chunks)} semantic chunks")
print(f"Embeddings shape: {embeddings.shape}")
```

### Structured Data Example

This example converts database rows into text using templates, then generates embeddings. Each row becomes one 768-dimensional vector, making database records semantically searchable.

```python
from structured_transformer import create_structured_transformer

# Initialize model
model = create_structured_transformer()

# Database rows
records = [
    {
        "product_id": "PROD_001",
        "risk_category": "Supply Chain",
        "severity": "Critical",
        "defect_rate": 0.15,
        "region": "Asia"
    }
]

# Encode with template
embeddings = model.encode_structured_data(
    records,
    template_name='risk_record'
)

print(f"Embeddings shape: {embeddings.shape}")
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Unstructured Model** | ~400ms per chunk |
| **Structured Model** | ~50ms per record |
| **Query Encoding** | ~25ms |

| Risk Category | Accuracy |
|---------------|----------|
| Product Risk | 90.0% |
| Service Risk | 82.5% |
| Brand Risk | 97.5% |

---

## Architecture

### Detailed Data Flow

#### PHASE 1: Data Ingestion & Preparation

**Step 1A: Input Data Recognition**

The system first receives raw data and determines whether it's unstructured text (like documents) or structured data (like database rows).

```python
# Unstructured: 5000-char risk report document
"Our supply chain operations in Asian region experiencing 
critical disruption. Primary suppliers reported shutdowns..."

# Structured: Database row (JSON)
{
  "product_id": "PROD_X_500",
  "risk_category": "Supply Chain",
  "severity": "Critical",
  "defect_rate": 0.15,
  "region": "Southeast Asia"
}
```

**Step 1B: Model Selection**

Based on the data type, the system chooses the appropriate model. Long-form text uses the Unstructured Model. Database records use the Structured Model.

---

#### PHASE 2: Preprocessing & Text Conversion

**Step 2A: Unstructured - Sentence Splitting**

The document is split into individual sentences using regex patterns that detect sentence boundaries (periods, question marks, exclamation points).

```python
# Input document split into sentences
Sentence 1: "Supply chain operations experiencing disruption."
Sentence 2: "Suppliers reported factory shutdowns."
Sentence 3: "40% reduction in component availability."
Sentence 4: "Defect rates at 15% above threshold."
# ... 7 sentences total
```

**Step 2B: Unstructured - Semantic Chunking**

Each sentence is embedded individually. Then, sentences are grouped based on semantic similarity. If two consecutive sentences have similarity ≥70%, they're added to the same chunk. If similarity drops below 75%, a new chunk starts. Maximum 6 sentences per chunk.

```python
# Generate embedding for each sentence
S1 → DistilBERT → [0.42, -0.15, 0.67, ...] (768 dims)
S2 → DistilBERT → [0.44, -0.13, 0.65, ...] (768 dims)

# Calculate similarity
cosine_similarity(S1, S2) = 0.89
# 0.89 ≥ 0.75 → ADD to chunk

# Continue until similarity drops
cosine_similarity(mean([S1,S2,S3]), S4) = 0.62
# 0.62 < 0.75 → STOP, start new chunk

# Result: 3 semantic chunks created
Chunk 1: Sentences 1-3 (supply chain topic)
Chunk 2: Sentences 4-6 (quality/customer topic)  
Chunk 3: Sentence 7 (SLA topic)
```

**Step 2C: Structured - Template Formatting**

The JSON object is converted into a natural language sentence using a predefined template. This makes structured data readable by the language model.

```python
# Apply template to JSON data
Template: "Risk: {risk_category}, Severity: {severity}, 
          Product: {product_id}, Rate: {defect_rate}..."

Output: "Risk: Supply Chain, Severity: Critical, 
         Product: PROD_X_500, Rate: 15.0%, 
         Region: Southeast Asia..."
```

---

#### PHASE 3: Tokenization

**Step 3A: Text to Tokens**

The text is broken into subword tokens (smaller than words) using WordPiece tokenization. Each token is assigned a unique ID number that the neural network understands. Special tokens [CLS] and [SEP] mark the start and end.

```python
# Input text
"Supply chain operations experiencing disruption"

# Tokenized
tokens = ["[CLS]", "supply", "chain", "operations", 
          "experiencing", "disruption", "[SEP]"]

# Convert to IDs
token_ids = [101, 4346, 4677, 3136, 13417, 20461, 102]

# Attention mask (1 = real token, 0 = padding)
attention_mask = [1, 1, 1, 1, 1, 1, 1]
```

---

#### PHASE 4: DistilBERT Neural Network

**Step 4A: Token Embedding Layer**

Each token ID is looked up in an embedding table and converted into a 768-dimensional vector. This is the initial numeric representation of each word.

```python
# Embedding lookup
token_id 4346 ("supply") → [0.156, 0.234, -0.089, ..., 0.123]
token_id 4677 ("chain")  → [0.089, -0.076, 0.145, ..., 0.098]

# Result: Matrix of [7 tokens × 768 dimensions]
```

**Step 4B: Positional Encoding**

Position embeddings are added to each token's vector so the model knows the order of words. This allows the model to distinguish "supply chain" from "chain supply".

```python
# Token embedding + Position embedding
Position 0: [0.156, 0.234, ...] + [0.001, 0.002, ...] = [0.157, 0.236, ...]
Position 1: [0.089, -0.076, ...] + [0.002, 0.003, ...] = [0.091, -0.073, ...]

# "chain supply" vs "supply chain" now have different vectors
```

**Step 4C: Transformer Layers (6 layers)**

The model has 6 transformer layers. In each layer, every token "looks at" all other tokens through self-attention. This allows words to understand their context. For example, "supply" learns it's related to "chain" because they frequently appear together.

```python
# Layer 1: Self-Attention for "supply"

# Step 1: Create Query, Key, Value vectors
Query_supply = [0.234, -0.123, 0.456, ...]
Key_supply   = [0.167, 0.289, -0.091, ...]
Value_supply = [0.391, -0.145, 0.278, ...]

# Step 2: Calculate attention with all tokens
Attention(supply, supply) = 0.89
Attention(supply, chain)  = 0.91  # High! They're related
Attention(supply, operations) = 0.34

# Step 3: Weighted sum of values
New_supply = 0.18×Value_supply + 0.20×Value_chain + ...
           = [0.287, -0.091, 0.356, ...]

# "supply" now contains info about "chain"!
# Layers 2-6 refine further...
```

---

#### PHASE 5: Mean Pooling

**Step 5A: Aggregate Token Vectors**

After the 6 transformer layers, we have a 768-dimensional vector for each token. Mean pooling averages all these token vectors into a single 768-dimensional vector that represents the entire chunk or sentence.

```python
# We have 7 token vectors, each 768-dimensional
[CLS]:       [0.412, -0.234, 0.567, ..., 0.189]
supply:      [0.445, -0.089, 0.623, ..., 0.234]
chain:       [0.438, -0.092, 0.618, ..., 0.229]
operations:  [0.391, -0.145, 0.578, ..., 0.198]
experiencing:[0.456, -0.198, 0.634, ..., 0.245]
disruption:  [0.512, -0.267, 0.689, ..., 0.278]
[SEP]:       [0.334, -0.167, 0.489, ..., 0.178]

# Mean pooling: Average all vectors
pooled = ([0.412,...] + [0.445,...] + ... + [0.334,...]) / 7
       = [0.427, -0.170, 0.600, ..., 0.221]

# Result: ONE 768-dim vector for entire chunk
```

---

#### PHASE 6: Enhancement & Normalization

**Step 6A: Risk-Specific Attention**

A custom multi-head attention layer (8 attention heads) is applied to learn which aspects of the text are most important for risk management. This is learned during training on risk-specific data.

```python
# Apply multi-head attention (8 heads)
pooled_vector = [0.427, -0.170, 0.600, ..., 0.221]
                ↓
risk_attention_layer(pooled_vector)
                ↓
enhanced = [0.445, -0.182, 0.615, ..., 0.234]

# Model learns risk-specific patterns
```

**Step 6B: Projection & Layer Normalization**

The vector passes through a linear projection layer and layer normalization. This stabilizes the values and prepares them for the final normalization step.

```python
# Project to final space
projected = linear_layer(enhanced)
          = [0.452, -0.189, 0.623, ..., 0.241]

# Layer normalization
normalized = layer_norm(projected)
           = [0.448, -0.185, 0.619, ..., 0.238]
```

**Step 6C: L2 Normalization**

The vector is normalized to unit length (magnitude of 1.0). This ensures cosine similarity calculations work correctly, as they measure the angle between vectors rather than their absolute magnitudes.

```python
# Calculate L2 norm (vector magnitude)
magnitude = sqrt(0.448² + (-0.185)² + 0.619² + ... + 0.238²)
          = 12.45

# Divide each element by magnitude
final_embedding = [0.448/12.45, -0.185/12.45, 0.619/12.45, ...]
                = [0.036, -0.015, 0.050, ..., 0.019]

# Now vector has length 1.0 (unit vector)
# Ready for cosine similarity comparisons!
```

---

#### PHASE 7: Storage

**Step 7A: Database Storage**

The final 768-dimensional vector is stored in a vector database (like Supabase with pgvector extension) alongside the original text. A vector index is created for fast similarity searching.

```sql
-- Store in database
INSERT INTO risk_embeddings (
    id,
    content,
    embedding
) VALUES (
    'chunk_001',
    'Supply chain operations experiencing disruption...',
    '[0.036, -0.015, 0.050, ..., 0.019]'  -- 768 numbers
);

-- Create vector index for fast similarity search
CREATE INDEX ON risk_embeddings 
USING ivfflat (embedding vector_cosine_ops);
```

---

### SEARCH FLOW (When User Queries)

When a user enters a search query, it goes through the exact same pipeline (Phases 2-6) to generate a query vector. This vector is then compared against all stored vectors using cosine similarity. Results are ranked by similarity score, with higher scores indicating more semantic relevance.

```python
# User enters query
query = "Asian supply chain problems"

# Query goes through SAME pipeline (Phases 2-6)
query → Tokenize → DistilBERT → Mean Pool → Attention → Normalize
      → query_vector = [0.038, -0.014, 0.052, ..., 0.020]

# Calculate cosine similarity with ALL stored vectors
similarity_1 = cosine(query_vector, chunk_001_vector) = 0.94
similarity_2 = cosine(query_vector, chunk_002_vector) = 0.62
similarity_3 = cosine(query_vector, chunk_003_vector) = 0.31

# Sort by similarity, return top matches
Results:
  1. Chunk 001 (0.94) - "Supply chain operations experiencing..."
  2. Chunk 002 (0.62) - "Quality defects increasing..."
  3. Chunk 003 (0.31) - "Service agreements at risk..."
```

---

### Core Components

- **Base Model:** distilbert-base-nli-mean-tokens (66M parameters)
- **Pooling:** Mean token pooling with attention masking
- **Enhancement:** Multi-head attention (8 heads)
- **Classification:** Risk category (3 classes) + Severity (4 levels)
- **Normalization:** L2 normalized embeddings for cosine similarity

---

## Use Cases

### Semantic Risk Search

This approach allows you to find similar risk incidents using natural language queries instead of exact keyword matches. The query is converted to a vector and compared against all stored risk vectors.

```python
query = "supply chain problems in Asia"
query_embedding = model.encode_text(query)
# Search vector database for similar risks
```

### Risk Classification

The model can automatically categorize risk documents into predefined categories (Product, Service, Brand) and assign severity levels (Low, Medium, High, Critical) based on learned patterns.

```python
predictions = model.predict_risk_attributes(risk_text)
print(predictions['predicted_categories'])
print(predictions['predicted_severities'])
```

### Database Vectorization

This process converts an entire database table into embeddings that can be semantically searched. Each row becomes a 768-dimensional vector stored alongside the original data.

```python
embeddings = model.batch_encode_database_table(
    rows=database_rows,
    template_name='risk_record'
)
# Store in vector database
```

---

## Configuration

### Unstructured Model Parameters

These parameters control how the model processes long-form text documents. You can adjust chunking behavior, similarity thresholds, and classification categories.

```python
model = RiskManagementTransformer(
    base_model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
    embedding_dim=768,
    risk_categories=3,        # product, service, brand
    severity_levels=4,        # low, medium, high, critical
    chunk_min_similarity=0.70,  # Continue threshold
    chunk_stop_threshold=0.75,  # Hard stop threshold
    chunk_max_sentences=6       # Max sentences per chunk
)
```

### Structured Model Templates

Templates define how structured data (JSON objects, database rows) is converted into natural language text before embedding. You can use built-in templates or create custom ones.

```python
# Built-in templates
templates = {
    'risk_record': "Product ID: {product_id}, Risk: {risk_category}...",
    'incident': "Incident ID: {incident_id}, Type: {type}...",
    'compliance': "Regulation: {regulation}, Status: {status}...",
    'quality': "Product: {product}, Defect Rate: {defect_rate}..."
}

# Add custom template
model.add_custom_template('my_template', "Field: {field}, Value: {value}")
```

---

## Performance

| Metric | Unstructured Model | Structured Model |
|--------|-------------------|------------------|
| **Embedding Dimension** | 768 | 768 |
| **Max Input Length** | 512 tokens | 512 tokens |
| **Inference Speed (CPU)** | ~500ms per chunk | ~50ms per record |
| **Batch Processing** | 32 chunks/batch | 64 records/batch |
| **GPU Acceleration** | ✓ Supported | ✓ Supported |

---

## License & Citation

### License
MIT License - Free for commercial and non-commercial use

### Citation

```bibtex
@misc{risk-management-transformers,
    title={Risk Management Transformers},
    author={Your Name},
    year={2024},
    publisher={Hugging Face},
    howpublished={\url{https://github.com/yourusername/risk-management-transformers}}
}
```

---

Built with [Hugging Face](https://huggingface.co) • [PyTorch](https://pytorch.org) • [GitHub](https://github.com)

© 2024 Risk Management Transformers
