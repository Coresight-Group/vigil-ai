# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VIGIL is an enterprise risk intelligence platform that combines dual AI sources (Claude AI + Grok Intelligence) with a unified DistilBERT embedding layer to analyze business risks across three data schema strategies.

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py

# Run with Docker
docker-compose up --build

# Run tests
pytest
```

The API runs on `http://localhost:5000` by default.

## Architecture

### Core Components (main.py)

- **DualPathRiskTransformer**: Main PyTorch module that embeds all data types into 768-dim vectors via DistilBERT, then routes through dual intelligence paths
- **GrokIntelligenceEngine**: Agentic RAG processor that calls X.AI's Grok API for industry context and best practices
- **RiskCorrelationEngine**: Detects self-conflicts, historical matches, and patterns via vector similarity
- **DocumentStore**: In-memory document storage with vector indexing

### Flask API (app.py)

Three validator classes implement different schema strategies:
- **StructuredValidator**: Schema-on-write, rejects invalid data
- **SemiStructuredValidator**: Hybrid schema, continues on partial errors
- **UnstructuredValidator**: Schema-on-read, graceful degradation

Key endpoints:
- `POST /api/risks/analyze` - Main analysis endpoint, routes to appropriate validator based on `data_type` field
- `GET /api/risks/search` - Semantic search across all stored vectors
- `GET /api/stats` - Vector store statistics
- `GET /api/health` - Health check

### Data Flow

```
Input (any data type)
    -> Validator (based on data_type)
    -> DistilBERT embedding (768-dim)
    -> Dual processing:
       1. Claude AI (Vigil Summary generation)
       2. Grok Intelligence (industry context)
    -> Complexity scoring (1-10)
    -> Alert generation (0-3 alerts based on complexity)
    -> Solution generation (1-7 tiered solutions)
```

### Configuration (config.py)

- `SCHEMA_STRATEGIES`: Defines validation behavior for each data type
- `ALERT_RULES`: Complexity-to-alert mapping
- `CONFIDENCE_LEVELS`: Baseline confidence by data type (structured: 95%, semi: 88%, unstructured: 80%)

## Environment Variables

Required in `.env`:
```
ANTHROPIC_API_KEY=sk-ant-xxxxx
XAI_API_KEY=xai_xxxxx
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=xxxxx
CLAUDE_MODEL=claude-opus-4-1
```

## Key Patterns

- All data types converge in the same 768-dim vector space via DistilBERT
- Confidence calculation varies by data type (see `calculate_confidence` in utils.py)
- Alerts scale with complexity: 1-3 complexity = 0 alerts, 4-7 = 1-2 alerts, 8+ = 2-3 alerts
- Solutions are tiered 1-4 based on timeline (immediate to strategic)

## Deployment

- Docker-based deployment via `Dockerfile` and `docker-compose.yml`
- Configured for HuggingFace Spaces via `app.yaml`
- Health checks on `/api/health` endpoint

## Auto-Updating Changelog

This repository uses a **pre-commit git hook** that automatically updates the Change Log below whenever you commit changes.

**How it works:**
1. When you run `git commit`, the hook detects staged files
2. It adds an entry to the Change Log table with date, files, and commit message
3. CLAUDE.md is automatically staged with your commit

**Location:** `.git/hooks/pre-commit`

**To bypass** (if needed): `git commit --no-verify`

---

## Change Log

| Date | File(s) Changed | Description | Impact |
|------|-----------------|-------------|--------|
| 2026-01-11 | CLAUDE.md | Initial creation | Provides Claude Code with project context for faster onboarding |
| 2026-01-11 | CLAUDE.md | Added Suggested Improvements section | Enables tracking of architectural and code enhancement opportunities |
| 2026-01-11 | .git/hooks/pre-commit | Added auto-changelog pre-commit hook | Automatically logs file changes to CLAUDE.md on every commit |

---

## Suggested Improvements

Prompts for code changes and architectural updates that would benefit VIGIL AI:

### Risk Management Enhancements

> **Add Risk Trend Analysis**
> Implement time-series tracking in `RiskCorrelationEngine` to detect escalating risk patterns over time. Add a `track_risk_trajectory()` method that compares current risk vectors against historical embeddings to identify worsening trends before they become critical.

> **Implement Risk Dependency Mapping**
> Create a graph-based dependency tracker in `main.py` that maps how risks cascade across business units. When a SUPPLY_CHAIN risk is detected, automatically identify downstream PRODUCTION and DELIVERY risks that may be affected.

> **Add Predictive Risk Scoring**
> Extend `_calculate_confidence()` to include a predictive component using historical match patterns. If similar past risks escalated, boost the current severity score proactively.

### Code Architecture Improvements

> **Implement Async Processing Pipeline**
> Convert the synchronous `forward()` method in `DualPathRiskTransformer` to async. Run Claude AI and Grok Intelligence calls in parallel using `asyncio.gather()` to reduce response latency by ~40%.

> **Add Caching Layer for Embeddings**
> Implement LRU caching for DistilBERT embeddings in `embed_text()`. Repeated or similar risk descriptions should retrieve cached vectors instead of recomputing, improving throughput for high-volume analysis.

> **Extract Validators to Separate Module**
> Move `StructuredValidator`, `SemiStructuredValidator`, and `UnstructuredValidator` from `app.py` to a dedicated `validators.py` module. This improves testability and separation of concerns.

> **Add Request Rate Limiting**
> Implement rate limiting middleware in `app.py` to protect the Grok and Claude API calls from quota exhaustion. Add exponential backoff for failed external API requests.

### Database & Storage Enhancements

> **Implement Persistent Vector Store**
> Replace in-memory `vector_store` list with pgvector in Supabase. Add `store_embedding()` and `query_similar()` methods that persist vectors across restarts and enable distributed deployments.

> **Add Audit Trail Table**
> Create a Supabase table to log all risk analyses with timestamps, user context, and decision outcomes. Enables compliance reporting and model performance tracking.

### Testing & Observability

> **Add Integration Tests for Dual-Path Flow**
> Create pytest fixtures that mock Claude and Grok responses, then test the complete analysis pipeline from validation through alert generation.

> **Implement Structured Logging**
> Replace print-style logging with structured JSON logs. Add correlation IDs to trace requests through the dual-path processing pipeline.
