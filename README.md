---
title: Risk Management Transformers
emoji: 🔍
colorFrom: Gray
colorTo: Yellow
sdk: static
pinned: false
---

# Risk Management Transformers

A dual-model architecture for structured and unstructured risk data. This repository contains specialized transformer models built on distilbert-base-nli-mean-tokens for risk management applications, enabling semantic search and risk classification across products, services, and brand reputation.

## Overview

This system provides two optimized models designed for different data types. The Unstructured Data Model uses semantic chunking with cosine similarity thresholds to intelligently process documents, reports, emails, and policy documents. The Structured Data Model uses template-based encoding for tabular data, processing database tables, JSON records, CSV data, and API responses.

The models generate 768-dimensional embeddings using PyTorch and Hugging Face transformers. Both models are optimized for risk management tasks including semantic search, automated classification, and database vectorization.

## Unstructured Data Model

The Unstructured Data Model is designed for long-form text processing. It excels at handling documents and reports, email threads, policy documents, news articles, and meeting notes. The model features intelligent sentence grouping with a 70% similarity threshold for continuing chunks and a 75% hard stop threshold for ending chunks. Each chunk contains a maximum of 6 sentences and uses mean pooling similarity for semantic analysis.

This model performs sentence-by-sentence analysis and is ideal for risk reports and incident narratives. The processing is slower than the structured model due to the semantic analysis required, but provides superior understanding of context and meaning in unstructured text.

## Structured Data Model

The Structured Data Model processes tabular data efficiently. It works best with database tables, JSON records, CSV data, API responses, and structured logs. The model includes pre-built templates for common data formats, supports custom formatting, enables batch processing for large datasets, uses field attention mechanisms, and automatically converts data types.

This model uses template-based formatting to convert structured data into natural language before embedding. Each database row becomes one embedding, making it much faster than the unstructured model. It's optimized for risk databases and compliance records.

## Model Comparison

The Unstructured Model accepts long-form text and documents as input. It uses semantic similarity-based chunking and performs sentence-by-sentence analysis. This model is optimal for risk reports and incident narratives but has slower performance due to the semantic analysis required.

The Structured Model accepts database rows and JSON objects as input. It doesn't use chunking, treating one row as one embedding. The processing method is template-based formatting. This model is optimal for risk databases and compliance records and offers faster performance through direct encoding.

## Installation

Clone the repository from GitHub. Navigate to the risk-management-transformers directory. Install the required dependencies including torch, transformers, sentence-transformers, and numpy using pip.

## Quick Start with Unstructured Data

To process unstructured data, first import and initialize the risk transformer model. Create a document containing your risk management text. The document might include information about supply chain disruptions in Asia, manufacturing delays affecting multiple product lines, quality control showing defects above threshold, and declining brand sentiment.

Call the chunk_and_embed method to process the document. The model will automatically split it into semantic chunks, analyzing sentence similarity and grouping related sentences together. The maximum is 6 sentences per chunk, and chunking stops when similarity drops below 75%. The result includes the created chunks and their corresponding embeddings, with each embedding being a 768-dimensional vector.

## Quick Start with Structured Data

To process structured data, import and initialize the structured transformer model. Define your database rows as dictionaries containing fields like product_id, risk_category, severity, defect_rate, and region.

Use the encode_structured_data method with a template name like 'risk_record'. The model converts each row into text using the template, then generates embeddings. Each row becomes one 768-dimensional vector, making database records semantically searchable. The resulting embeddings can be stored in a vector database for similarity search.

## Performance Metrics

The Unstructured Model processes chunks at approximately 400 milliseconds per chunk. The Structured Model processes records at approximately 50 milliseconds per record. Query encoding takes approximately 25 milliseconds regardless of model type.

Classification accuracy varies by risk category. Product Risk classification achieves 90.0% accuracy. Service Risk classification achieves 82.5% accuracy. Brand Risk classification achieves 97.5% accuracy.

## Architecture and Data Flow

The system processes data through seven distinct phases. Understanding this flow helps you optimize performance and troubleshoot issues.

### Phase 1: Data Ingestion and Preparation

The system first receives raw data and determines whether it's unstructured text like documents or structured data like database rows. For unstructured data, you might have a 5000-character risk report document describing supply chain operations experiencing critical disruption with primary suppliers reporting shutdowns.

For structured data, you have a database row in JSON format containing fields like product_id, risk_category, severity, defect_rate, and region. The system examines the input and chooses the appropriate model. Long-form text uses the Unstructured Model while database records use the Structured Model.

### Phase 2: Preprocessing and Text Conversion

For unstructured data, the document is split into individual sentences using regex patterns that detect sentence boundaries like periods, question marks, and exclamation points. A typical document might split into sentences about supply chain operations experiencing disruption, suppliers reporting factory shutdowns, reduction in component availability, and defect rates above threshold.

Each sentence is then embedded individually. Sentences are grouped based on semantic similarity. If two consecutive sentences have similarity of 70% or higher, they're added to the same chunk. If similarity drops below 75%, a new chunk starts. The maximum is 6 sentences per chunk. The result might be three semantic chunks covering supply chain topics, quality and customer topics, and SLA topics.

For structured data, the JSON object is converted into a natural language sentence using a predefined template. This makes structured data readable by the language model. A template might format the data as "Risk: Supply Chain, Severity: Critical, Product: PROD_X_500, Rate: 15.0%, Region: Southeast Asia" and so on.

### Phase 3: Tokenization

The text is broken into subword tokens, which are smaller units than words, using WordPiece tokenization. Each token is assigned a unique ID number that the neural network understands. Special tokens [CLS] and [SEP] mark the start and end of the sequence.

For example, the text "Supply chain operations experiencing disruption" becomes tokens [CLS], supply, chain, operations, experiencing, disruption, and [SEP]. These are converted to IDs like 101, 4346, 4677, 3136, 13417, 20461, and 102. An attention mask is created where 1 indicates a real token and 0 indicates padding.

### Phase 4: DistilBERT Neural Network Processing

Each token ID is looked up in an embedding table and converted into a 768-dimensional vector. This is the initial numeric representation of each word. For instance, token ID 4346 representing "supply" becomes a vector with 768 numbers, and token ID 4677 representing "chain" becomes another 768-dimensional vector. The result is a matrix with dimensions of 7 tokens by 768 dimensions.

Position embeddings are then added to each token's vector so the model knows the order of words. This allows the model to distinguish "supply chain" from "chain supply". Each position gets a small embedding added to the token embedding, making the final representation position-aware.

The model has 6 transformer layers. In each layer, every token looks at all other tokens through self-attention. This allows words to understand their context. For example, "supply" learns it's related to "chain" because they frequently appear together. The self-attention mechanism creates Query, Key, and Value vectors for each token, calculates attention scores between all pairs of tokens, and produces weighted combinations. High attention between "supply" and "chain" means the model recognizes they form a meaningful phrase. Layers 2 through 6 refine these representations further.

### Phase 5: Mean Pooling

After the 6 transformer layers, we have a 768-dimensional vector for each token. Mean pooling averages all these token vectors into a single 768-dimensional vector that represents the entire chunk or sentence. 

For our 7 tokens, we have vectors for [CLS], supply, chain, operations, experiencing, disruption, and [SEP]. Mean pooling averages all these vectors together by adding them up and dividing by 7. The result is one 768-dimensional vector for the entire chunk.

### Phase 6: Enhancement and Normalization

A custom multi-head attention layer with 8 attention heads is applied to learn which aspects of the text are most important for risk management. This is learned during training on risk-specific data. The pooled vector passes through this risk attention layer and becomes enhanced with risk-specific patterns.

The vector then passes through a linear projection layer and layer normalization. This stabilizes the values and prepares them for the final normalization step. The projection transforms the vector into the final embedding space, and layer normalization ensures stable training and inference.

Finally, the vector is normalized to unit length with a magnitude of 1.0. This ensures cosine similarity calculations work correctly, as they measure the angle between vectors rather than their absolute magnitudes. The L2 norm is calculated by taking the square root of the sum of squared elements. Each element is then divided by this magnitude, creating a unit vector ready for cosine similarity comparisons.

### Phase 7: Storage

The final 768-dimensional vector is stored in a vector database like Supabase with pgvector extension alongside the original text. A vector index is created for fast similarity searching. The database stores the chunk ID, the original content text, and the embedding as an array of 768 numbers. An index using ivfflat or similar algorithm enables fast approximate nearest neighbor search.

### Search Flow

When a user enters a search query, it goes through the exact same pipeline from phases 2 through 6 to generate a query vector. This vector is then compared against all stored vectors using cosine similarity. Results are ranked by similarity score, with higher scores indicating more semantic relevance.

For example, the query "Asian supply chain problems" is tokenized, processed through DistilBERT, mean pooled, enhanced with attention, and normalized into a query vector. This query vector is compared with every stored chunk vector using cosine similarity. A chunk about supply chain operations might have 0.94 similarity, a chunk about quality defects might have 0.62 similarity, and a chunk about service agreements might have 0.31 similarity. Results are sorted by similarity score and the top matches are returned.

## Core Components

The base model is distilbert-base-nli-mean-tokens with 66 million parameters. It uses mean token pooling with attention masking to create sentence-level embeddings. Enhancement comes from multi-head attention with 8 heads for risk-specific learning. Classification includes risk category prediction with 3 classes for product, service, and brand risks, plus severity level prediction with 4 levels for low, medium, high, and critical. All embeddings are L2 normalized for cosine similarity comparisons.

## Use Cases

Semantic Risk Search allows you to find similar risk incidents using natural language queries instead of exact keyword matches. The query is converted to a vector and compared against all stored risk vectors. You can search for "supply chain problems in Asia" and find semantically similar incidents even if they use different wording.

Risk Classification enables the model to automatically categorize risk documents into predefined categories like Product, Service, or Brand. It also assigns severity levels like Low, Medium, High, or Critical based on patterns learned during training. This automates the triage and categorization process for incoming risk reports.

Database Vectorization converts an entire database table into embeddings that can be semantically searched. Each row becomes a 768-dimensional vector stored alongside the original data. This enables natural language search over structured data, allowing queries like "show me critical risks in Asia" to find relevant database records.

## Configuration

For the Unstructured Model, you can configure several parameters. The base model name defaults to sentence-transformers/distilbert-base-nli-mean-tokens. The embedding dimension is 768. You can set the number of risk categories, defaulting to 3 for product, service, and brand. Severity levels default to 4 for low, medium, high, and critical. The chunk minimum similarity threshold defaults to 0.70 for continuing chunks. The chunk stop threshold defaults to 0.75 for hard stops. The maximum sentences per chunk defaults to 6.

For the Structured Model, templates define how structured data is converted into natural language text before embedding. You can use built-in templates or create custom ones. Built-in templates include risk_record for general risk data, incident for incident reports, compliance for regulatory compliance records, and quality for quality control data. You can add custom templates by defining a template string with field placeholders.

## Performance Specifications

Both models use 768-dimensional embeddings. The maximum input length is 512 tokens for both models. Inference speed on CPU is approximately 500 milliseconds per chunk for the Unstructured Model and approximately 50 milliseconds per record for the Structured Model. Batch processing handles 32 chunks per batch for the Unstructured Model and 64 records per batch for the Structured Model. GPU acceleration is supported for both models.

## License and Citation

This project is released under the MIT License, making it free for commercial and non-commercial use. When citing this work in research or publications, please reference it as Risk Management Transformers by the author, published in 2024 on Hugging Face.

## Technical Stack

Built with Hugging Face transformers library for model architecture and pretrained weights. Uses PyTorch as the deep learning framework. Hosted and shared through GitHub for version control and collaboration.

Copyright 2024 Risk Management Transformers
