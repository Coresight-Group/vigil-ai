---
title: CoreSightGroup
colorFrom: gray
colorTo: yellow
sdk: static
pinned: false
---

# CoreSightGroup

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Risk Management Transformers | Hugging Face</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
            background: #0a0a0a;
            color: #ffffff;
            line-height: 1.8;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 60px 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 80px;
            padding: 60px 0;
            border-bottom: 1px solid #d4af37;
        }
        
        .logo {
            font-size: 3em;
            margin-bottom: 20px;
            color: #d4af37;
            font-weight: 100;
            letter-spacing: 2px;
        }
        
        h1 {
            font-size: 3em;
            font-weight: 300;
            margin-bottom: 20px;
            color: #ffffff;
            letter-spacing: -1px;
        }
        
        .tagline {
            font-size: 1.2em;
            color: #888;
            font-weight: 300;
            margin-top: 20px;
        }
        
        .badges {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 30px;
            flex-wrap: wrap;
        }
        
        .badge {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 0.85em;
            color: #d4af37;
            letter-spacing: 1px;
        }
        
        .section {
            margin-bottom: 80px;
        }
        
        h2 {
            font-size: 2em;
            font-weight: 300;
            margin-bottom: 30px;
            color: #d4af37;
            border-left: 4px solid #d4af37;
            padding-left: 20px;
        }
        
        h3 {
            font-size: 1.5em;
            font-weight: 300;
            margin: 40px 0 20px 0;
            color: #ffffff;
        }
        
        p {
            color: #bbb;
            margin-bottom: 20px;
            font-size: 1em;
        }
        
        .model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 40px;
            margin: 40px 0;
        }
        
        .model-card {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 40px;
            border-radius: 4px;
            transition: all 0.3s ease;
        }
        
        .model-card:hover {
            border-color: #d4af37;
            transform: translateY(-5px);
        }
        
        .model-card h3 {
            color: #d4af37;
            margin-top: 0;
            font-size: 1.8em;
        }
        
        .model-icon {
            font-size: 3em;
            margin-bottom: 20px;
            opacity: 0.8;
        }
        
        .code-block {
            background: #0a0a0a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 30px;
            margin: 30px 0;
            overflow-x: auto;
            font-family: 'SF Mono', monospace;
            font-size: 0.9em;
            line-height: 1.6;
        }
        
        .code-block code {
            color: #d4af37;
        }
        
        .comment {
            color: #666;
        }
        
        .string {
            color: #888;
        }
        
        .keyword {
            color: #fff;
        }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 40px 0;
            background: #1a1a1a;
        }
        
        .comparison-table th {
            background: #0a0a0a;
            color: #d4af37;
            padding: 20px;
            text-align: left;
            font-weight: 300;
            border: 1px solid #333;
        }
        
        .comparison-table td {
            padding: 20px;
            border: 1px solid #333;
            color: #bbb;
        }
        
        .comparison-table tr:hover {
            background: #0f0f0f;
        }
        
        .feature-list {
            list-style: none;
            margin: 20px 0;
        }
        
        .feature-list li {
            padding: 15px 0;
            padding-left: 30px;
            position: relative;
            color: #bbb;
        }
        
        .feature-list li:before {
            content: "→";
            position: absolute;
            left: 0;
            color: #d4af37;
            font-weight: bold;
        }
        
        .install-box {
            background: #1a1a1a;
            border-left: 4px solid #d4af37;
            padding: 30px;
            margin: 30px 0;
            border-radius: 0 4px 4px 0;
        }
        
        .install-box h3 {
            margin-top: 0;
            color: #d4af37;
        }
        
        .divider {
            height: 1px;
            background: #333;
            margin: 60px 0;
        }
        
        .architecture-box {
            background: #0a0a0a;
            border: 1px solid #333;
            padding: 40px;
            margin: 40px 0;
            text-align: center;
        }
        
        .flow-diagram {
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin: 40px 0;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .flow-box {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 20px 30px;
            min-width: 150px;
            text-align: center;
            font-size: 0.9em;
        }
        
        .flow-arrow {
            color: #d4af37;
            font-size: 2em;
        }
        
        .footer {
            text-align: center;
            margin-top: 100px;
            padding: 40px 0;
            border-top: 1px solid #333;
            color: #666;
        }
        
        .link {
            color: #d4af37;
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: border-color 0.3s;
        }
        
        .link:hover {
            border-bottom-color: #d4af37;
        }
        
        @media (max-width: 768px) {
            .model-grid {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 2em;
            }
            
            .logo {
                font-size: 2.5em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Risk Management Transformers</h1>
            <p class="tagline">Dual-Model Architecture for Structured & Unstructured Risk Data</p>
            
            <div class="badges">
                <span class="badge">DistilBERT-base</span>
                <span class="badge">768-dim embeddings</span>
                <span class="badge">PyTorch</span>
                <span class="badge">Hugging Face</span>
            </div>
        </div>
        
        <!-- Overview -->
        <div class="section">
            <h2>Overview</h2>
            <p>
                A specialized transformer architecture built on <code>distilbert-base-nli-mean-tokens</code> for risk management applications. 
                This repository contains two optimized models designed for different data types, enabling semantic search and risk classification 
                across products, services, and brand reputation.
            </p>
        </div>
        
        <!-- Models -->
        <div class="section">
            <h2>Models</h2>
            
            <div class="model-grid">
                <!-- Unstructured Model -->
                <div class="model-card">
                    <div class="model-icon">📄</div>
                    <h3>Unstructured Data Model</h3>
                    <p style="color: #888; margin-bottom: 30px;">
                        Semantic chunking with cosine similarity thresholds
                    </p>
                    
                    <h4 style="color: #fff; font-weight: 300; margin: 20px 0 10px 0;">Best For:</h4>
                    <ul class="feature-list">
                        <li>Documents and reports</li>
                        <li>Email threads</li>
                        <li>Policy documents</li>
                        <li>News articles</li>
                        <li>Meeting notes</li>
                    </ul>
                    
                    <h4 style="color: #fff; font-weight: 300; margin: 20px 0 10px 0;">Features:</h4>
                    <ul class="feature-list">
                        <li>Intelligent sentence grouping</li>
                        <li>70% similarity threshold</li>
                        <li>75% hard stop threshold</li>
                        <li>Max 6 sentences per chunk</li>
                        <li>Mean pooling similarity</li>
                    </ul>
                </div>
                
                <!-- Structured Model -->
                <div class="model-card">
                    <div class="model-icon">📊</div>
                    <h3>Structured Data Model</h3>
                    <p style="color: #888; margin-bottom: 30px;">
                        Template-based encoding for tabular data
                    </p>
                    
                    <h4 style="color: #fff; font-weight: 300; margin: 20px 0 10px 0;">Best For:</h4>
                    <ul class="feature-list">
                        <li>Database tables</li>
                        <li>JSON records</li>
                        <li>CSV data</li>
                        <li>API responses</li>
                        <li>Structured logs</li>
                    </ul>
                    
                    <h4 style="color: #fff; font-weight: 300; margin: 20px 0 10px 0;">Features:</h4>
                    <ul class="feature-list">
                        <li>Pre-built templates</li>
                        <li>Custom formatting</li>
                        <li>Batch processing</li>
                        <li>Field attention</li>
                        <li>Auto type conversion</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="divider"></div>
        
        <!-- Comparison Table -->
        <div class="section">
            <h2>Model Comparison</h2>
            
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Unstructured Model</th>
                        <th>Structured Model</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Input Type</td>
                        <td>Long-form text, documents</td>
                        <td>Database rows, JSON objects</td>
                    </tr>
                    <tr>
                        <td>Chunking Strategy</td>
                        <td>Semantic similarity-based</td>
                        <td>None (1 row = 1 embedding)</td>
                    </tr>
                    <tr>
                        <td>Processing Method</td>
                        <td>Sentence-by-sentence analysis</td>
                        <td>Template-based formatting</td>
                    </tr>
                    <tr>
                        <td>Optimal Use Case</td>
                        <td>Risk reports, incident narratives</td>
                        <td>Risk databases, compliance records</td>
                    </tr>
                    <tr>
                        <td>Performance</td>
                        <td>Slower (semantic analysis)</td>
                        <td>Faster (direct encoding)</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="divider"></div>
        
        <!-- Installation -->
        <div class="section">
            <h2>Installation</h2>
            
            <div class="code-block">
<span class="comment"># Clone repository</span>
<span class="keyword">git clone</span> https://github.com/yourusername/risk-management-transformers.git
<span class="keyword">cd</span> risk-management-transformers

<span class="comment"># Install dependencies</span>
<span class="keyword">pip install</span> torch transformers sentence-transformers numpy
            </div>
        </div>
        
        <!-- Quick Start -->
        <div class="section">
            <h2>Quick Start</h2>
            
            <h3>Unstructured Data Example</h3>
            <div class="code-block">
<span class="keyword">from</span> risk_transformer <span class="keyword">import</span> create_risk_transformer

<span class="comment"># Initialize model</span>
model = create_risk_transformer()

<span class="comment"># Process long document with semantic chunking</span>
document = <span class="string">"""
Supply chain disruptions in Asia continue. Manufacturing 
delays affect multiple product lines. Quality control shows 
defects above threshold. Brand sentiment declining.
"""</span>

<span class="comment"># Chunk and embed</span>
chunks, embeddings = model.chunk_and_embed(document)

<span class="keyword">print</span>(<span class="string">f"Created {len(chunks)} semantic chunks"</span>)
<span class="keyword">print</span>(<span class="string">f"Embeddings shape: {embeddings.shape}"</span>)
            </div>
            
            <h3>Structured Data Example</h3>
            <div class="code-block">
<span class="keyword">from</span> structured_transformer <span class="keyword">import</span> create_structured_transformer

<span class="comment"># Initialize model</span>
model = create_structured_transformer()

<span class="comment"># Database rows</span>
records = [
    {
        <span class="string">"product_id"</span>: <span class="string">"PROD_001"</span>,
        <span class="string">"risk_category"</span>: <span class="string">"Supply Chain"</span>,
        <span class="string">"severity"</span>: <span class="string">"Critical"</span>,
        <span class="string">"defect_rate"</span>: 0.15,
        <span class="string">"region"</span>: <span class="string">"Asia"</span>
    }
]

<span class="comment"># Encode with template</span>
embeddings = model.encode_structured_data(
    records,
    template_name=<span class="string">'risk_record'</span>
)

<span class="keyword">print</span>(<span class="string">f"Embeddings shape: {embeddings.shape}"</span>)
            </div>
        </div>
        
        <div class="divider"></div>
        
        <!-- Architecture -->
        <div class="section">
            <h2>Architecture</h2>
            
            <div class="architecture-box">
                <h3 style="color: #d4af37; margin-bottom: 40px;">Data Flow</h3>
                
                <div class="flow-diagram">
                    <div class="flow-box">Input Data</div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-box">Preprocessing</div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-box">DistilBERT</div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-box">Mean Pooling</div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-box">Attention Layer</div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-box">768-dim Vector</div>
                </div>
            </div>
            
        <!-- Architecture -->
        <div class="section">
            <h2>Architecture</h2>
            
            <h3 style="margin-top: 50px;">Detailed Data Flow</h3>
            
            <div style="background: #1a1a1a; border: 1px solid #333; padding: 40px; border-radius: 4px; margin: 30px 0;">
                <h4 style="color: #d4af37; font-weight: 300; margin-bottom: 20px;">PHASE 1: Data Ingestion & Preparation</h4>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 1A: Input Data Recognition</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        System receives raw data and identifies type (structured vs unstructured).
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># Unstructured: 5000-char risk report document</span>
<span class="string">"Our supply chain operations in Asian region experiencing 
critical disruption. Primary suppliers reported shutdowns..."</span>

<span class="comment"># Structured: Database row (JSON)</span>
{
  <span class="string">"product_id"</span>: <span class="string">"PROD_X_500"</span>,
  <span class="string">"risk_category"</span>: <span class="string">"Supply Chain"</span>,
  <span class="string">"severity"</span>: <span class="string">"Critical"</span>,
  <span class="string">"defect_rate"</span>: 0.15,
  <span class="string">"region"</span>: <span class="string">"Southeast Asia"</span>
}
                    </div>
                </div>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 1B: Model Selection</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        Decision: Long-form text? → Unstructured Model. Database record? → Structured Model.
                    </p>
                </div>
            </div>
            
            <div style="background: #1a1a1a; border: 1px solid #333; padding: 40px; border-radius: 4px; margin: 30px 0;">
                <h4 style="color: #d4af37; font-weight: 300; margin-bottom: 20px;">PHASE 2: Preprocessing & Text Conversion</h4>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 2A: Unstructured - Sentence Splitting</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        Document broken into sentences using regex patterns.
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># Input document split into sentences</span>
Sentence 1: <span class="string">"Supply chain operations experiencing disruption."</span>
Sentence 2: <span class="string">"Suppliers reported factory shutdowns."</span>
Sentence 3: <span class="string">"40% reduction in component availability."</span>
Sentence 4: <span class="string">"Defect rates at 15% above threshold."</span>
<span class="comment"># ... 7 sentences total</span>
                    </div>
                </div>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 2B: Unstructured - Semantic Chunking</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        Sentences grouped by semantic similarity (≥70% threshold, &lt;75% stops, max 6 sentences).
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># Generate embedding for each sentence</span>
S1 → DistilBERT → [0.42, -0.15, 0.67, ...] (768 dims)
S2 → DistilBERT → [0.44, -0.13, 0.65, ...] (768 dims)

<span class="comment"># Calculate similarity</span>
cosine_similarity(S1, S2) = 0.89
<span class="comment"># 0.89 ≥ 0.75 → ADD to chunk</span>

<span class="comment"># Continue until similarity drops</span>
cosine_similarity(mean([S1,S2,S3]), S4) = 0.62
<span class="comment"># 0.62 &lt; 0.75 → STOP, start new chunk</span>

<span class="comment"># Result: 3 semantic chunks created</span>
Chunk 1: Sentences 1-3 (supply chain topic)
Chunk 2: Sentences 4-6 (quality/customer topic)  
Chunk 3: Sentence 7 (SLA topic)
                    </div>
                </div>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 2C: Structured - Template Formatting</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        JSON converted to natural language using templates.
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># Apply template to JSON data</span>
Template: <span class="string">"Risk: {risk_category}, Severity: {severity}, 
          Product: {product_id}, Rate: {defect_rate}..."</span>

Output: <span class="string">"Risk: Supply Chain, Severity: Critical, 
         Product: PROD_X_500, Rate: 15.0%, 
         Region: Southeast Asia..."</span>
                    </div>
                </div>
            </div>
            
            <div style="background: #1a1a1a; border: 1px solid #333; padding: 40px; border-radius: 4px; margin: 30px 0;">
                <h4 style="color: #d4af37; font-weight: 300; margin-bottom: 20px;">PHASE 3: Tokenization</h4>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 3A: Text to Tokens</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        Text broken into subword tokens using WordPiece tokenization.
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># Input text</span>
<span class="string">"Supply chain operations experiencing disruption"</span>

<span class="comment"># Tokenized</span>
tokens = [<span class="string">"[CLS]"</span>, <span class="string">"supply"</span>, <span class="string">"chain"</span>, <span class="string">"operations"</span>, 
          <span class="string">"experiencing"</span>, <span class="string">"disruption"</span>, <span class="string">"[SEP]"</span>]

<span class="comment"># Convert to IDs</span>
token_ids = [101, 4346, 4677, 3136, 13417, 20461, 102]

<span class="comment"># Attention mask (1 = real token, 0 = padding)</span>
attention_mask = [1, 1, 1, 1, 1, 1, 1]
                    </div>
                </div>
            </div>
            
            <div style="background: #1a1a1a; border: 1px solid #333; padding: 40px; border-radius: 4px; margin: 30px 0;">
                <h4 style="color: #d4af37; font-weight: 300; margin-bottom: 20px;">PHASE 4: DistilBERT Neural Network</h4>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 4A: Token Embedding Layer</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        Each token ID converted to 768-dimensional vector.
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># Embedding lookup</span>
token_id 4346 (<span class="string">"supply"</span>) → [0.156, 0.234, -0.089, ..., 0.123]
token_id 4677 (<span class="string">"chain"</span>)  → [0.089, -0.076, 0.145, ..., 0.098]

<span class="comment"># Result: Matrix of [7 tokens × 768 dimensions]</span>
                    </div>
                </div>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 4B: Positional Encoding</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        Position information added so model knows word order.
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># Token embedding + Position embedding</span>
Position 0: [0.156, 0.234, ...] + [0.001, 0.002, ...] = [0.157, 0.236, ...]
Position 1: [0.089, -0.076, ...] + [0.002, 0.003, ...] = [0.091, -0.073, ...]

<span class="comment"># "chain supply" vs "supply chain" now have different vectors</span>
                    </div>
                </div>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 4C: Transformer Layers (6 layers)</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        Self-attention allows each token to look at ALL other tokens and refine its meaning.
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># Layer 1: Self-Attention for "supply"</span>

<span class="comment">Step 1: Create Query, Key, Value vectors</span>
Query_supply = [0.234, -0.123, 0.456, ...]
Key_supply   = [0.167, 0.289, -0.091, ...]
Value_supply = [0.391, -0.145, 0.278, ...]

<span class="comment">Step 2: Calculate attention with all tokens</span>
Attention(supply, supply) = 0.89
Attention(supply, chain)  = 0.91  <span class="comment"># High! They're related</span>
Attention(supply, operations) = 0.34

<span class="comment">Step 3: Weighted sum of values</span>
New_supply = 0.18×Value_supply + 0.20×Value_chain + ...
           = [0.287, -0.091, 0.356, ...]

<span class="comment"># "supply" now contains info about "chain"!</span>
<span class="comment"># Layers 2-6 refine further...</span>
                    </div>
                </div>
            </div>
            
            <div style="background: #1a1a1a; border: 1px solid #333; padding: 40px; border-radius: 4px; margin: 30px 0;">
                <h4 style="color: #d4af37; font-weight: 300; margin-bottom: 20px;">PHASE 5: Mean Pooling</h4>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 5A: Aggregate Token Vectors</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        All token vectors averaged into ONE 768-dim vector per chunk.
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># We have 7 token vectors, each 768-dimensional</span>
[CLS]:       [0.412, -0.234, 0.567, ..., 0.189]
supply:      [0.445, -0.089, 0.623, ..., 0.234]
chain:       [0.438, -0.092, 0.618, ..., 0.229]
operations:  [0.391, -0.145, 0.578, ..., 0.198]
experiencing:[0.456, -0.198, 0.634, ..., 0.245]
disruption:  [0.512, -0.267, 0.689, ..., 0.278]
[SEP]:       [0.334, -0.167, 0.489, ..., 0.178]

<span class="comment"># Mean pooling: Average all vectors</span>
pooled = ([0.412,...] + [0.445,...] + ... + [0.334,...]) / 7
       = [0.427, -0.170, 0.600, ..., 0.221]

<span class="comment"># Result: ONE 768-dim vector for entire chunk</span>
                    </div>
                </div>
            </div>
            
            <div style="background: #1a1a1a; border: 1px solid #333; padding: 40px; border-radius: 4px; margin: 30px 0;">
                <h4 style="color: #d4af37; font-weight: 300; margin-bottom: 20px;">PHASE 6: Enhancement & Normalization</h4>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 6A: Risk-Specific Attention</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        Multi-head attention learns which features matter most for risk management.
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># Apply multi-head attention (8 heads)</span>
pooled_vector = [0.427, -0.170, 0.600, ..., 0.221]
                ↓
risk_attention_layer(pooled_vector)
                ↓
enhanced = [0.445, -0.182, 0.615, ..., 0.234]

<span class="comment"># Model learns risk-specific patterns</span>
                    </div>
                </div>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 6B: Projection & Layer Normalization</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        Final projection and normalization for stability.
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># Project to final space</span>
projected = linear_layer(enhanced)
          = [0.452, -0.189, 0.623, ..., 0.241]

<span class="comment"># Layer normalization</span>
normalized = layer_norm(projected)
           = [0.448, -0.185, 0.619, ..., 0.238]
                    </div>
                </div>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 6C: L2 Normalization</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        Make vector unit length for cosine similarity.
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># Calculate L2 norm (vector magnitude)</span>
magnitude = sqrt(0.448² + (-0.185)² + 0.619² + ... + 0.238²)
          = 12.45

<span class="comment"># Divide each element by magnitude</span>
final_embedding = [0.448/12.45, -0.185/12.45, 0.619/12.45, ...]
                = [0.036, -0.015, 0.050, ..., 0.019]

<span class="comment"># Now vector has length 1.0 (unit vector)</span>
<span class="comment"># Ready for cosine similarity comparisons!</span>
                    </div>
                </div>
            </div>
            
            <div style="background: #1a1a1a; border: 1px solid #333; padding: 40px; border-radius: 4px; margin: 30px 0;">
                <h4 style="color: #d4af37; font-weight: 300; margin-bottom: 20px;">PHASE 7: Storage</h4>
                
                <div style="margin: 30px 0;">
                    <strong style="color: #fff;">Step 7A: Database Storage</strong>
                    <p style="margin: 15px 0 15px 20px; color: #bbb;">
                        Final 768-dim vector stored in vector database (e.g., Supabase pgvector).
                    </p>
                    
                    <div class="code-block" style="margin: 15px 0 15px 20px;">
<span class="comment"># Store in database</span>
INSERT INTO risk_embeddings (
    id,
    content,
    embedding
) VALUES (
    <span class="string">'chunk_001'</span>,
    <span class="string">'Supply chain operations experiencing disruption...'</span>,
    <span class="string">'[0.036, -0.015, 0.050, ..., 0.019]'</span>  <span class="comment"># 768 numbers</span>
);

<span class="comment"># Create vector index for fast similarity search</span>
CREATE INDEX ON risk_embeddings 
USING ivfflat (embedding vector_cosine_ops);
                    </div>
                </div>
            </div>
            
            <div style="background: #0a0a0a; border: 2px solid #d4af37; padding: 40px; border-radius: 4px; margin: 30px 0;">
                <h4 style="color: #d4af37; font-weight: 300; margin-bottom: 20px;">SEARCH FLOW (When User Queries)</h4>
                
                <div class="code-block">
<span class="comment"># User enters query</span>
query = <span class="string">"Asian supply chain problems"</span>

<span class="comment"># Query goes through SAME pipeline (Phases 2-6)</span>
query → Tokenize → DistilBERT → Mean Pool → Attention → Normalize
      → query_vector = [0.038, -0.014, 0.052, ..., 0.020]

<span class="comment"># Calculate cosine similarity with ALL stored vectors</span>
similarity_1 = cosine(query_vector, chunk_001_vector) = 0.94
similarity_2 = cosine(query_vector, chunk_002_vector) = 0.62
similarity_3 = cosine(query_vector, chunk_003_vector) = 0.31

<span class="comment"># Sort by similarity, return top matches</span>
Results:
  1. Chunk 001 (0.94) - "Supply chain operations experiencing..."
  2. Chunk 002 (0.62) - "Quality defects increasing..."
  3. Chunk 003 (0.31) - "Service agreements at risk..."
                </div>
            </div>
            
            <h3>Core Components</h3>
            <ul class="feature-list">
                <li><strong>Base Model:</strong> distilbert-base-nli-mean-tokens (66M parameters)</li>
                <li><strong>Pooling:</strong> Mean token pooling with attention masking</li>
                <li><strong>Enhancement:</strong> Multi-head attention (8 heads)</li>
                <li><strong>Classification:</strong> Risk category (3 classes) + Severity (4 levels)</li>
                <li><strong>Normalization:</strong> L2 normalized embeddings for cosine similarity</li>
            </ul>
        </div>
        
        <div class="divider"></div>
        
        <!-- Use Cases -->
        <div class="section">
            <h2>Use Cases</h2>
            
            <div class="install-box">
                <h3>Semantic Risk Search</h3>
                <p>Find similar risk incidents across your database using natural language queries.</p>
                <div class="code-block" style="margin-top: 20px;">
query = <span class="string">"supply chain problems in Asia"</span>
query_embedding = model.encode_text(query)
<span class="comment"># Search vector database for similar risks</span>
                </div>
            </div>
            
            <div class="install-box">
                <h3>Risk Classification</h3>
                <p>Automatically categorize and assess severity of risk documents.</p>
                <div class="code-block" style="margin-top: 20px;">
predictions = model.predict_risk_attributes(risk_text)
<span class="keyword">print</span>(predictions[<span class="string">'predicted_categories'</span>])
<span class="keyword">print</span>(predictions[<span class="string">'predicted_severities'</span>])
                </div>
            </div>
            
            <div class="install-box">
                <h3>Database Vectorization</h3>
                <p>Convert entire risk databases into searchable vector embeddings.</p>
                <div class="code-block" style="margin-top: 20px;">
embeddings = model.batch_encode_database_table(
    rows=database_rows,
    template_name=<span class="string">'risk_record'</span>
)
<span class="comment"># Store in Supabase with pgvector</span>
                </div>
            </div>
        </div>
        
        <div class="divider"></div>
        
        <!-- Performance Graphs -->
        <div class="section">
            <h2>Performance Metrics</h2>
            
            <h3>Embedding Generation Speed</h3>
            <div class="architecture-box" style="padding: 60px 40px;">
                <svg width="100%" height="350" viewBox="0 0 800 350">
                    <!-- Grid lines -->
                    <line x1="100" y1="280" x2="750" y2="280" stroke="#333" stroke-width="1"/>
                    <line x1="100" y1="230" x2="750" y2="230" stroke="#222" stroke-width="1" stroke-dasharray="5,5"/>
                    <line x1="100" y1="180" x2="750" y2="180" stroke="#222" stroke-width="1" stroke-dasharray="5,5"/>
                    <line x1="100" y1="130" x2="750" y2="130" stroke="#222" stroke-width="1" stroke-dasharray="5,5"/>
                    <line x1="100" y1="80" x2="750" y2="80" stroke="#222" stroke-width="1" stroke-dasharray="5,5"/>
                    
                    <!-- Y-axis labels -->
                    <text x="70" y="285" fill="#666" font-size="12">0ms</text>
                    <text x="50" y="235" fill="#666" font-size="12">100ms</text>
                    <text x="50" y="185" fill="#666" font-size="12">200ms</text>
                    <text x="50" y="135" fill="#666" font-size="12">300ms</text>
                    <text x="50" y="85" fill="#666" font-size="12">400ms</text>
                    
                    <!-- Bars - Unstructured Model -->
                    <rect x="200" y="80" width="120" height="200" fill="#d4af37" opacity="0.8"/>
                    <text x="260" y="310" fill="#d4af37" font-size="14" text-anchor="middle">Unstructured</text>
                    <text x="260" y="330" fill="#d4af37" font-size="12" text-anchor="middle">Model</text>
                    <text x="260" y="65" fill="#d4af37" font-size="16" text-anchor="middle" font-weight="bold">400ms</text>
                    
                    <!-- Bars - Structured Model -->
                    <rect x="380" y="255" width="120" height="25" fill="#888" opacity="0.8"/>
                    <text x="440" y="310" fill="#888" font-size="14" text-anchor="middle">Structured</text>
                    <text x="440" y="330" fill="#888" font-size="12" text-anchor="middle">Model</text>
                    <text x="440" y="245" fill="#888" font-size="16" text-anchor="middle" font-weight="bold">50ms</text>
                    
                    <!-- Bars - Query -->
                    <rect x="560" y="267.5" width="120" height="12.5" fill="#fff" opacity="0.6"/>
                    <text x="620" y="310" fill="#fff" font-size="14" text-anchor="middle">Query</text>
                    <text x="620" y="330" fill="#fff" font-size="12" text-anchor="middle">Encoding</text>
                    <text x="620" y="258" fill="#fff" font-size="16" text-anchor="middle" font-weight="bold">25ms</text>
                    
                    <!-- Title -->
                    <text x="425" y="35" fill="#bbb" font-size="18" text-anchor="middle">Average Processing Time per Item</text>
                </svg>
            </div>
            
            <h3>Model Accuracy by Risk Category</h3>
            <div class="architecture-box" style="padding: 60px 40px;">
                <svg width="100%" height="350" viewBox="0 0 800 350">
                    <!-- Grid lines -->
                    <line x1="100" y1="280" x2="750" y2="280" stroke="#333" stroke-width="2"/>
                    <line x1="100" y1="230" x2="750" y2="230" stroke="#222" stroke-width="1" stroke-dasharray="5,5"/>
                    <line x1="100" y1="180" x2="750" y2="180" stroke="#222" stroke-width="1" stroke-dasharray="5,5"/>
                    <line x1="100" y1="130" x2="750" y2="130" stroke="#222" stroke-width="1" stroke-dasharray="5,5"/>
                    <line x1="100" y1="80" x2="750" y2="80" stroke="#222" stroke-width="1" stroke-dasharray="5,5"/>
                    
                    <!-- Y-axis labels -->
                    <text x="70" y="285" fill="#666" font-size="13">0%</text>
                    <text x="55" y="235" fill="#666" font-size="13">25%</text>
                    <text x="55" y="185" fill="#666" font-size="13">50%</text>
                    <text x="55" y="135" fill="#666" font-size="13">75%</text>
                    <text x="50" y="85" fill="#666" font-size="13">100%</text>
                    
                    <!-- Product Risk -->
                    <rect x="200" y="100" width="120" height="180" fill="#d4af37" opacity="0.8"/>
                    <text x="260" y="310" fill="#d4af37" font-size="14" text-anchor="middle">Product</text>
                    <text x="260" y="330" fill="#d4af37" font-size="12" text-anchor="middle">Risk</text>
                    <text x="260" y="85" fill="#d4af37" font-size="16" text-anchor="middle" font-weight="bold">90%</text>
                    
                    <!-- Service Risk -->
                    <rect x="380" y="115" width="120" height="165" fill="#d4af37" opacity="0.7"/>
                    <text x="440" y="310" fill="#d4af37" font-size="14" text-anchor="middle">Service</text>
                    <text x="440" y="330" fill="#d4af37" font-size="12" text-anchor="middle">Risk</text>
                    <text x="440" y="100" fill="#d4af37" font-size="16" text-anchor="middle" font-weight="bold">82.5%</text>
                    
                    <!-- Brand Risk -->
                    <rect x="560" y="85" width="120" height="195" fill="#d4af37" opacity="0.9"/>
                    <text x="620" y="310" fill="#d4af37" font-size="14" text-anchor="middle">Brand</text>
                    <text x="620" y="330" fill="#d4af37" font-size="12" text-anchor="middle">Risk</text>
                    <text x="620" y="70" fill="#d4af37" font-size="16" text-anchor="middle" font-weight="bold">97.5%</text>
                    
                    <!-- Title -->
                    <text x="425" y="35" fill="#bbb" font-size="18" text-anchor="middle">Classification Accuracy</text>
                </svg>
            </div>
        </div>
        
        <div class="divider"></div>
        
        <!-- Configuration -->
        <div class="section">
            <h2>Configuration</h2>
            
            <h3>Unstructured Model Parameters</h3>
            <div class="code-block">
model = RiskManagementTransformer(
    base_model_name=<span class="string">"sentence-transformers/distilbert-base-nli-mean-tokens"</span>,
    embedding_dim=768,
    risk_categories=3,        <span class="comment"># product, service, brand</span>
    severity_levels=4,        <span class="comment"># low, medium, high, critical</span>
    chunk_min_similarity=0.70,  <span class="comment"># Continue threshold</span>
    chunk_stop_threshold=0.75,  <span class="comment"># Hard stop threshold</span>
    chunk_max_sentences=6       <span class="comment"># Max sentences per chunk</span>
)
            </div>
            
            <h3>Structured Model Templates</h3>
            <div class="code-block">
<span class="comment"># Built-in templates</span>
templates = {
    <span class="string">'risk_record'</span>: <span class="string">"Product ID: {product_id}, Risk: {risk_category}..."</span>,
    <span class="string">'incident'</span>: <span class="string">"Incident ID: {incident_id}, Type: {type}..."</span>,
    <span class="string">'compliance'</span>: <span class="string">"Regulation: {regulation}, Status: {status}..."</span>,
    <span class="string">'quality'</span>: <span class="string">"Product: {product}, Defect Rate: {defect_rate}..."</span>
}

<span class="comment"># Add custom template</span>
model.add_custom_template(<span class="string">'my_template'</span>, <span class="string">"Field: {field}, Value: {value}"</span>)
            </div>
        </div>
        
        <div class="divider"></div>
        
        <!-- Performance -->
        <div class="section">
            <h2>Performance</h2>
            
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Unstructured Model</th>
                        <th>Structured Model</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Embedding Dimension</td>
                        <td>768</td>
                        <td>768</td>
                    </tr>
                    <tr>
                        <td>Max Input Length</td>
                        <td>512 tokens</td>
                        <td>512 tokens</td>
                    </tr>
                    <tr>
                        <td>Inference Speed (CPU)</td>
                        <td>~500ms per chunk</td>
                        <td>~50ms per record</td>
                    </tr>
                    <tr>
                        <td>Batch Processing</td>
                        <td>32 chunks/batch</td>
                        <td>64 records/batch</td>
                    </tr>
                    <tr>
                        <td>GPU Acceleration</td>
                        <td>✓ Supported</td>
                        <td>✓ Supported</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="divider"></div>
        
        <!-- License -->
        <div class="section">
            <h2>License & Citation</h2>
            
            <div class="install-box">
                <h3>License</h3>
                <p>MIT License - Free for commercial and non-commercial use</p>
            </div>
            
            <div class="install-box">
                <h3>Citation</h3>
                <div class="code-block" style="margin-top: 20px;">
@misc{risk-management-transformers,
    title={Risk Management Transformers},
    author={Your Name},
    year={2024},
    publisher={Hugging Face},
    howpublished={\url{https://github.com/yourusername/risk-management-transformers}}
}
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>
                Built with <a href="https://huggingface.co" class="link">Hugging Face</a> • 
                <a href="https://pytorch.org" class="link">PyTorch</a> • 
                <a href="https://github.com" class="link">GitHub</a>
            </p>
            <p style="margin-top: 20px; font-size: 0.9em;">
                © 2024 Risk Management Transformers
            </p>
        </div>
    </div>
</body>
</html>
