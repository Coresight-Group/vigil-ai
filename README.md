 ---
title: VIGIL Risk Intelligence
emoji: ⚠️
colorFrom: gray
colorTo: gray
sdk: docker
app_file: app.py
app_port: 5000
license: apache-2.0
inference: false
library_name: transformers
pipeline_tag: text-classification
tags:
  - risk-management
  - risk-detection
  - security
  - machine-learning
  - enterprise
  - dual-source
  - schema-on-read
  - schema-on-write
  - structured-data
  - unstructured-data
  - semi-structured-data
  - vigil-summary
  - alerts
  - solutions
  - claude-ai
  - anthropic
  - grok
  - agentic-rag
  - distilbert
  - semantic-search
  - vector-database
  - supabase
  - pgvector
  - flask
  - docker
  - transformers
---

# ⚠️ VIGIL: Enterprise Risk Intelligence System

## Executive Summary

VIGIL is an enterprise-grade risk intelligence platform that transforms how companies detect and respond to business threats. Unlike traditional risk management systems, VIGIL combines three distinct data processing paths with their own schema strategies:

1. **Structured Data (Schema on Write)** - PostgreSQL/Supabase database with strict schema validation
2. **Semi-Structured Data (Hybrid Schema)** - JSONB fields with flexible metadata and validation at boundaries
3. **Unstructured Data (Schema on Read)** - Natural language processing with graceful validation

All three data types flow through a unified **DistilBERT embedding layer** (768-dimensional vectors) that converts them into a common semantic space. Then:
- **Claude AI (Anthropic)** provides deep reasoning, synthesis, and Vigil Summary generation
- **Grok Intelligence (X.AI)** acts as an Agentic RAG engine, processing first-party data for industry knowledge and context

The system achieves 95%+ confidence when both paths agree, with complete attribution and transparency throughout.

---

## 🏗️ Architecture Overview

### Three Data Processing Paths

#### **Path 1: Structured Data (Schema on Write)**

**Data Characteristics:**
- Fixed schema in PostgreSQL/Supabase
- Strict validation at write time
- High consistency and reliability
- Examples: Risk records, alerts, solutions, audit logs

**Schema Definition:**
```sql
risks (
  id UUID PRIMARY KEY,
  organization_id UUID,
  description VARCHAR(5000) NOT NULL,
  risk_type VARCHAR(50),
  severity ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'),
  complexity_score INTEGER (1-10),
  confidence DECIMAL (0-1),
  created_at TIMESTAMP,
  updated_at TIMESTAMP
)

risk_metadata JSONB (
  source: string,
  category_hint: string,
  tags: array,
  custom_fields: object
)
```

**Processing Flow:**
1. Input validation (Schema on Write) - REJECT if invalid
2. Store in PostgreSQL with validation
3. DistilBERT vectorization (768-dim)
4. pgvector storage and indexing
5. Structured analysis (pattern matching, correlations)
6. Output: Verified historical data, proven solutions

**Components:**
- SchemaValidator - Database schema enforcement
- DocumentStore - Persistent storage
- Vector Indexing - pgvector semantic search
- RiskCorrelationEngine - Pattern detection

---

#### **Path 2: Semi-Structured Data (Hybrid Schema)**

**Data Characteristics:**
- Flexible structure with core required fields
- Validation at read/write boundaries
- JSONB storage for extensibility
- Examples: Risk metadata, classifications, custom attributes

**Schema Definition:**
```json
{
  "id": "uuid",
  "risk_type": "SUPPLY_CHAIN|QUALITY|DELIVERY|PRODUCTION|BRAND",
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "attributes": {
    "personnel_impact": "string (optional)",
    "financial_impact": "number (optional)",
    "timeline": "string (optional)",
    "custom_field_1": "any (optional)"
  },
  "classifications": {
    "industry": "string",
    "business_unit": "string",
    "priority_level": "integer"
  }
}
```

**Processing Flow:**
1. Flexible input (Core fields required, extras optional)
2. Store with required fields in structured columns
3. Optional/custom fields in JSONB
4. DistilBERT vectorization (768-dim)
5. Dual processing:
   - Structured fields → Database queries
   - JSONB fields → Semantic search via vectors
6. Hybrid analysis (both structured & semantic)
7. Output: Complete picture with flexibility

**Components:**
- FlexibleValidator - Hybrid validation
- JSONB Storage - PostgreSQL JSON support
- Vector Embeddings - All data types vectorized
- Hybrid QueryEngine - Structured + semantic queries

---

#### **Path 3: Unstructured Data (Schema on Read)**

**Data Characteristics:**
- Natural language, free-form text
- Validation at read time only
- Graceful degradation on errors
- Examples: Risk descriptions, Grok intelligence, Claude analysis

**Processing Flow:**
1. Accept any text (20-5000 chars)
2. Store as-is (no validation rejection)
3. DistilBERT vectorization (768-dim)
4. Claude AI natural language processing:
   - Deep reasoning and understanding
   - Vigil Summary generation (seamlessly blended)
   - Risk contextualization
   - Confidence validation
5. Grok Intelligence (Agentic RAG):
   - First-party data analysis
   - Industry knowledge extraction
   - Real-time context generation
   - Best practice research
6. Validation at read time (Schema on Read):
   - Extract structured insights from unstructured text
   - Validate confidence and reasoning
   - Handle errors gracefully
7. Output: Insights with reasoning, industry context

**Components:**
- PromptValidator - Flexible input validation
- DistilBERT Processing - All data types vectorized
- Claude AI - Deep reasoning engine
- GrokIntelligenceEngine - Agentic RAG processor
- ResponseValidator - Schema on Read validation

---

### Unified Embedding Layer

**DistilBERT (768-dimensional vectors)**

All three data paths converge at the embedding layer:

```
Structured Data                Semi-Structured              Unstructured Data
     ↓                                ↓                           ↓
Database Records            Flexible JSON Records        Natural Language Text
     ↓                                ↓                           ↓
     └────────────────┬───────────────┬───────────────┬─────────────┘
                      │
                      ↓
              DistilBERT Tokenizer
              (converts all to tokens)
                      │
                      ↓
              DistilBERT Model
              (generates 768-dim vectors)
                      │
                      ├─→ pgvector (semantic search)
                      ├─→ Cosine similarity (pattern matching)
                      └─→ Unified semantic space
                      
              Output: All data in same vector space
```

**Key Benefits:**
- Semantic search across all data types
- Pattern matching works on all structures
- Cross-structure relationships discoverable
- Unified similarity scoring
- Foundation for Claude AI synthesis

---

### Dual Intelligence Processing

#### **Claude AI (Deep Reasoning)**
Processes all data types for:
- **Vigil Summary**: Seamlessly blends organizational + industry data
- **Deep Reasoning**: Why conclusions matter, confidence levels
- **Risk Contextualization**: Places risks in business context
- **Synthesis**: Combines findings from all paths into coherent narrative
- **Confidence Validation**: Explains why findings are reliable

#### **Grok Intelligence (Agentic RAG)**
Acts as Agentic RAG processor:
- **First-Party Data Analysis**: Deep analysis of company's own risk data
- **Industry Knowledge Synthesis**: Connects company data to industry trends
- **Real-Time Context**: Current market situation and comparisons
- **Best Practice Research**: What others did, how it worked
- **Pattern Discovery**: Emerges from first-party data through intelligent analysis

---

### Complete System Architecture

```
DUAL DATA INPUTS
├─ Structured: "Risk type: Supply Chain, Severity: HIGH"
├─ Semi-Structured: {"description": "...", "custom": {...}}
└─ Unstructured: "Our supplier in region X is experiencing..."

        │
        ↓

DATA INGESTION LAYER
├─ Path 1: SchemaValidator (Structured, Schema on Write)
│          ├─ Validate schema
│          ├─ REJECT if invalid
│          └─ Store in PostgreSQL
│
├─ Path 2: FlexibleValidator (Semi-Structured, Hybrid)
│          ├─ Require core fields
│          ├─ Accept optional fields
│          ├─ Store structured + JSONB
│          └─ Continue on partial errors
│
└─ Path 3: PromptValidator (Unstructured, Schema on Read)
           ├─ Accept all text
           ├─ Store as-is
           ├─ Validate later
           └─ Graceful degradation

        │
        ↓

UNIFIED EMBEDDING LAYER (DistilBERT)
├─ Tokenize all data types
├─ Generate 768-dimensional vectors
├─ Index in pgvector
└─ Create semantic relationships

        │
        ↓

DUAL INTELLIGENCE ENGINES
├─ Claude AI (Deep Reasoning)
│  ├─ Analyze all vector representations
│  ├─ Generate Vigil Summary
│  ├─ Explain reasoning
│  └─ Validate confidence
│
└─ Grok Intelligence (Agentic RAG)
   ├─ Process first-party structured data
   ├─ Extract industry patterns
   ├─ Research market context
   ├─ Discover best practices
   └─ Cross-reference with vectors

        │
        ↓

SYNTHESIS & ANALYSIS
├─ Pattern Recognition (all vectors)
├─ Historical Matching (structured + vectors)
├─ Self-Conflict Detection (unstructured analysis)
├─ Cascading Effects (structured relationships)
├─ Timeline Correlations (structured + semi)
└─ Industry Validation (Grok findings)

        │
        ↓

VIGIL SUMMARY GENERATION
├─ Claude AI: Seamlessly blended narrative
├─ No explicit source labeling
├─ Complete provenance (internal)
├─ Integrated with industry knowledge
└─ Confidence levels and reasoning

        │
        ↓

ALERT & SOLUTION GENERATION
├─ Complexity Scoring (1-10, all paths)
├─ Alert Triggering (0-3, auto-triggered)
├─ Solution Ranking (1-7, with tiers)
├─ Timeline Estimation (all sources)
└─ Success Probability (proven + research)

        │
        ↓

RESPONSE WITH VALIDATION
├─ Vigil Summary (unified narrative)
├─ Risk Classification
├─ Alerts (0-3)
├─ Solutions (1-7)
├─ Detailed Analysis
└─ Validation Info (all 3 paths)
```

---

## 🎯 Data Type Handling

### Structured Data (Schema on Write)

**Validation Strategy:** STRICT
- All fields required
- Types enforced at write time
- Invalid data REJECTED before storage
- 100% data quality guarantee

**Processing:**
```
Input → SchemaValidator → 
  ├─ Type check ✓
  ├─ Required fields ✓
  ├─ Length validation ✓
  └─ Reject if invalid ✗
        │
        ↓
    PostgreSQL Storage
        │
        ↓
    DistilBERT Vectorization
        │
        ↓
    Pattern Matching & Analysis
        │
        ↓
Output: High-confidence structured insights
```

**Examples:**
- Database risk records
- Alert definitions
- Solution templates
- Metadata schemas

---

### Semi-Structured Data (Hybrid Schema)

**Validation Strategy:** FLEXIBLE
- Core fields required
- Optional fields accepted
- Validation at boundaries
- Graceful handling of unknowns

**Processing:**
```
Input → FlexibleValidator →
  ├─ Required fields check ✓
  ├─ Type hints for optional
  ├─ Store structured + JSONB
  └─ Continue on partial match
        │
        ↓
    PostgreSQL Storage
    ├─ Structured columns
    └─ JSONB for flexibility
        │
        ↓
    DistilBERT Vectorization
    (entire record including JSONB)
        │
        ↓
    Dual Processing
    ├─ Structured queries
    ├─ Semantic similarity
    └─ Hybrid analysis
        │
        ↓
Output: Complete picture with flexibility
```

**Examples:**
- Risk classifications
- Custom attributes
- Business context
- Flexible metadata

---

### Unstructured Data (Schema on Read)

**Validation Strategy:** GRACEFUL
- Accept all text
- No rejection at write time
- Validation during reading/analysis
- Degradation on errors

**Processing:**
```
Input → PromptValidator (length only) →
  ├─ Accept text (20-5000 chars)
  ├─ Store as-is
  └─ Continue on any content
        │
        ↓
    DistilBERT Vectorization
        │
        ├─ Claude AI Processing
        │  ├─ Deep reasoning
        │  ├─ Vigil Summary
        │  ├─ Confidence validation
        │  └─ Graceful degradation
        │
        └─ Grok Intelligence (Agentic RAG)
           ├─ First-party data analysis
           ├─ Industry context
           ├─ Best practices
           └─ Real-time knowledge
        │
        ↓
    ResponseValidator (Schema on Read)
    ├─ Extract structure from analysis
    ├─ Validate findings
    ├─ Handle errors gracefully
    └─ Continue with partial results
        │
        ↓
Output: Insights with reasoning & context
```

**Examples:**
- Risk descriptions
- Incident reports
- Real-time intelligence
- Natural language queries

---

## ✨ Key Features

### 1. Data Type Agnostic Processing
- **DistilBERT**: All data types → 768-dimensional vectors
- **Semantic Space**: All data comparable and searchable
- **Pattern Recognition**: Works across data types
- **Unified Analysis**: Structured + semi + unstructured together

### 2. Schema Strategy by Data Type
- **Structured (Schema on Write)**: Enforce at write, guaranteed quality
- **Semi-Structured (Hybrid)**: Flexible core, structured boundaries
- **Unstructured (Schema on Read)**: Graceful validation during reading

### 3. Vigil Summary (Seamlessly Blended)
- Claude AI synthesizes all three data types
- No explicit labeling ("this is from DB vs Grok")
- Appears as unified, integrated analysis
- Complete provenance tracking (internal)
- Full reasoning and confidence explanation

### 4. Agentic RAG with Grok Intelligence
- **First-Party Data Processing**: Deep analysis of company's own data
- **Industry Knowledge Synthesis**: Connects to broader context
- **Real-Time Context**: Current market situation
- **Best Practice Research**: What works and why
- **Intelligent Analysis**: Agent asks questions, discovers patterns

### 5. Complexity-Based Alerts (0-3)
- Complexity 1-3: 0 alerts (routine)
- Complexity 4-7: 1-2 alerts (review)
- Complexity 8+: 2-3 alerts (crisis)
- Auto-triggered based on analysis

### 6. Hierarchical Solutions (1-7)
- **Tier 1**: Immediate actions (1 hour)
- **Tier 2**: Urgent actions (1-2 days)
- **Tier 3**: Critical path (3-7 days)
- **Tier 4**: Strategic (3+ weeks)
- Ranked by success probability

---

## 🔌 API Endpoints

### Health Check
```
GET /api/health
```

### Analyze Risk (All Data Types)
```
POST /api/risks/analyze
Content-Type: application/json

{
  "description": "Risk description (20-5000 chars)",
  "data_type": "structured|semi-structured|unstructured",
  "metadata": {
    "risk_type": "SUPPLY_CHAIN",
    "severity": "HIGH",
    "custom_field": "value"
  }
}
```

**Response:**
```json
{
  "success": true,
  "classification": {
    "risk_type": "SUPPLY_CHAIN",
    "severity": "HIGH",
    "confidence": 0.95,
    "complexity_score": 7
  },
  "vigil_summary": {
    "current_situation": "...",
    "industry_context": "...",
    "proven_approaches": "..."
  },
  "alerts": [
    {
      "alert_level": "HIGH",
      "title": "...",
      "recommendation": "..."
    }
  ],
  "solutions": [
    {
      "tier": 1,
      "title": "...",
      "success_probability": 0.85
    }
  ],
  "validation_info": {
    "data_type_processed": "structured|semi-structured|unstructured",
    "schema_strategy": "schema-on-write|hybrid|schema-on-read",
    "paths_used": ["structured", "unstructured"],
    "structured_confidence": 0.95,
    "unstructured_confidence": 0.85,
    "consensus_level": "high|medium|low"
  }
}
```

### Semantic Search
```
GET /api/risks/search?q=supply+chain&top_k=5&data_types=all
```

### Get Statistics
```
GET /api/stats
```

---

## 📁 Project Files

| File | Purpose | Lines |
|------|---------|-------|
| main.py | Dual-path transformer + Grok Agentic RAG | 702+ |
| app.py | Flask API + Schema validators | 600+ |
| README.md | Documentation | This file |

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
```
ANTHROPIC_API_KEY=sk-ant-xxxxx
XAI_API_KEY=xai_xxxxx
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=xxxxx
FLASK_ENV=development
CLAUDE_MODEL=claude-opus-4-1
```

### Running
```bash
python app.py
```

API available at `http://localhost:5000`

---

## 🔐 Security

- Structured data: Validated at write (Schema on Write)
- Semi-structured: Validated at boundaries (Hybrid)
- Unstructured: Validated at read (Schema on Read)
- All credentials in .env (git ignored)
- No hardcoded API keys
- Complete audit trail

---

## ✅ Features Implemented

✅ Structured data (Schema on Write) - PostgreSQL/Supabase
✅ Semi-structured data (Hybrid Schema) - JSONB fields
✅ Unstructured data (Schema on Read) - Natural language
✅ DistilBERT embeddings (768-dim) for all data types
✅ Claude AI (deep reasoning + Vigil Summary)
✅ Grok Intelligence (Agentic RAG processor)
✅ Unified semantic space (pgvector indexing)
✅ Pattern recognition across all types
✅ Complexity-based alerts (0-3)
✅ Hierarchical solutions (1-7, 4 tiers)
✅ Full source attribution
✅ Graceful error handling
✅ Comprehensive validation strategies
✅ Flask REST API
✅ Docker deployment ready
✅ GitHub + HuggingFace ready

---

**VIGIL: Enterprise Risk Intelligence at Scale** ⚠️

Protecting organizations with AI-powered risk detection and mitigation.