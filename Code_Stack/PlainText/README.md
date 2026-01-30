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
  - multi-tenant
  - schema-on-read
  - schema-on-write
  - structured-data
  - unstructured-data
  - semi-structured-data
  - vigil-summary
  - alerts
  - solutions
  - grok
  - xai
  - agentic-rag
  - distilbert
  - semantic-search
  - vector-database
  - supabase
  - pgvector
  - flask
  - docker
  - transformers
  - gunicorn
  - nginx
---

# VIGIL: Enterprise Risk Intelligence Platform

## Executive Summary

VIGIL is an enterprise-grade risk intelligence platform that transforms how organizations detect, analyze, and respond to business threats. The system uniquely handles three distinct data types through different validation strategies and synthesizes insights from dual AI sources.

**Core Architecture:**
- **Three Data Paths**: Structured (Schema on Write), Semi-Structured (Hybrid), Unstructured (Schema on Read)
- **Unified Embedding Layer**: DistilBERT 768-dimensional vectors for semantic search
- **Grok Intelligence**: Powered by Grok (X.AI) for deep reasoning, industry context, and Agentic RAG
- **Multi-Tenant**: Subdomain-based client isolation with dedicated databases

**Key Outcomes:**
- 95%+ confidence when both AI paths agree
- Complexity-based alerts (0-3)
- Hierarchical solutions (1-7 across 4 tiers)
- Complete attribution and audit trails

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Python Flask REST API |
| Frontend | HTML5, CSS3, JavaScript |
| Database | PostgreSQL (Supabase) - Master + Client |
| Vector Store | pgvector (768-dim embeddings) |
| Embedding Model | DistilBERT |
| AI Intelligence | Grok 3.0 (X.AI) - Deep Reasoning & Agentic RAG |
| Deployment | Docker + Gunicorn + Nginx |
| ML Framework | PyTorch, Transformers |

---

## Project Structure

```
Vigil_AI/
├── Model_Code/
│   ├── Backend/
│   │   ├── app.py                    # Flask API with validators
│   │   ├── main.py                   # Dual-path transformer & Grok engine
│   │   ├── config.py                 # Application configuration
│   │   ├── auth_config.py            # Authentication & multi-tenant setup
│   │   ├── utils.py                  # Helper functions
│   │   ├── data_sync_service.py      # Data synchronization
│   │   ├── supabase-schema-master.sql # Master DB (clients, auth, admin)
│   │   ├── supabase-schema-client.sql # Client DB (risks, documents, etc.)
│   │   ├── requirements.txt          # Python dependencies
│   │   └── deployment/
│   │       ├── gunicorn.conf.py      # WSGI server config
│   │       ├── nginx.conf            # Reverse proxy config
│   │       └── hostinger/
│   │           ├── setup.sh          # VPS deployment script
│   │           └── vigil.service     # Systemd service file
│   │
│   ├── FrontEnd/
│   │   ├── Interface/                # Main dashboard
│   │   │   ├── index.html            # Risk analysis UI
│   │   │   ├── script.js             # Chat & search functionality
│   │   │   ├── styles.css            # Main styling
│   │   │   └── admin-styles.css      # Admin panel styling
│   │   └── Login/                    # Authentication portal
│   │       ├── index.html            # Dual login (User/Admin)
│   │       ├── script.js             # Login logic
│   │       └── styles.css            # Login styling
│   │
│   ├── Docker/
│   │   ├── Dockerfile                # Container image
│   │   └── docker-compose.yml        # Service orchestration
│   │
│   ├── Keys_Security/                # Credentials (git-ignored)
│   ├── PlainText/
│   │   └── README.md                 # This file
│   └── .env.example                  # Environment template
│
└── .gitignore                        # Git ignore rules
```

---

## Three-Schema Architecture

VIGIL processes data through three distinct validation strategies, each optimized for different data characteristics:

### Path 1: Structured Data (Schema on Write)

**Validation Strategy:** STRICT - Reject invalid data before storage

```sql
risks (
  id UUID PRIMARY KEY,
  organization_id UUID,
  description VARCHAR(5000) NOT NULL,
  risk_type VARCHAR(50),
  severity ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'),
  complexity_score INTEGER (1-10),
  confidence DECIMAL (0-1)
)
```

**Processing Flow:**
1. Validate schema at write time
2. REJECT if invalid
3. Store in PostgreSQL
4. Vectorize with DistilBERT (768-dim)
5. Index in pgvector
6. Pattern matching & analysis

**Use Cases:** Risk records, alert definitions, solution templates

---

### Path 2: Semi-Structured Data (Hybrid Schema)

**Validation Strategy:** FLEXIBLE - Core fields required, extras accepted

```json
{
  "id": "uuid",
  "risk_type": "SUPPLY_CHAIN|QUALITY|DELIVERY|PRODUCTION|BRAND",
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "attributes": {
    "personnel_impact": "string (optional)",
    "financial_impact": "number (optional)",
    "custom_field": "any (optional)"
  }
}
```

**Processing Flow:**
1. Validate required fields
2. Accept optional/custom fields
3. Store structured columns + JSONB
4. Vectorize entire record
5. Dual processing (structured queries + semantic search)
6. Hybrid analysis

**Use Cases:** Risk metadata, custom attributes, classifications

---

### Path 3: Unstructured Data (Schema on Read)

**Validation Strategy:** GRACEFUL - Accept all, validate during analysis

**Processing Flow:**
1. Accept text (20-5000 chars)
2. Store as-is (no rejection)
3. Vectorize with DistilBERT
4. Grok AI deep reasoning & Vigil Summary
5. Grok Intelligence context & Agentic RAG
6. Validate at read time
7. Graceful degradation on errors

**Use Cases:** Risk descriptions, incident reports, natural language queries

---

## Grok AI Intelligence

### Grok (X.AI) - Complete AI Engine

VIGIL is powered entirely by Grok, providing both deep reasoning and Agentic RAG capabilities:

**Deep Reasoning & Analysis:**
- **Vigil Summary**: Seamlessly blended narrative (no explicit source labeling)
- **Risk Contextualization**: Places risks in business context
- **Confidence Validation**: Explains why findings are reliable
- **Synthesis**: Combines all data paths into coherent analysis

**Agentic RAG & Industry Intelligence:**
- **First-Party Analysis**: Deep analysis of organization's own data
- **Industry Knowledge**: Connects to broader market context
- **Real-Time Context**: Current situation and comparisons
- **Best Practices**: What others did, how it worked
- **Pattern Discovery**: Emerges from intelligent analysis

**Conversational Features:**
- Natural dialogue for follow-up questions
- Industry-specific responses with data references in parentheses
- Proactive follow-up question suggestions when analyzing alerts

---

## API Endpoints

### Health & Status
```
GET /api/health              # System health check
GET /api/stats               # Statistics & storage info
```

### Risk Analysis
```
POST /api/risks/analyze      # Analyze risk (any data type)
GET  /api/risks/search       # Semantic search
GET  /api/risks/<id>         # Get specific risk
GET  /api/risks/recent       # Recent analyses
```

### Procurement
```
POST /api/procurement/suggest           # Get suggestions
GET  /api/procurement/history           # View history
PUT  /api/procurement/<id>/status       # Update status
```

### Enterprise
```
GET /api/suppliers           # Supplier database
GET /api/inventory           # Inventory management
GET /api/documents/search    # Cross-document search
```

### Analysis Request Example
```json
POST /api/risks/analyze
{
  "description": "Our primary supplier in Southeast Asia is experiencing severe flooding affecting 3 manufacturing facilities",
  "data_type": "unstructured",
  "metadata": {
    "risk_type": "SUPPLY_CHAIN",
    "severity": "HIGH"
  }
}
```

### Response Structure
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
    "situation": "...",
    "context": "...",
    "approach": "...",
    "timeline": "..."
  },
  "alerts": [
    {
      "alert_level": "HIGH",
      "title": "Supply Chain Disruption Imminent",
      "recommendation": "..."
    }
  ],
  "solutions": [
    {
      "tier": 1,
      "urgency": "IMMEDIATE",
      "title": "...",
      "success_probability": 0.85
    }
  ],
  "validation_info": {
    "data_type_processed": "unstructured",
    "schema_strategy": "schema-on-read",
    "grok_integrated": true,
    "grok_processed": true
  }
}
```

---

## Alert & Solution Generation

### Complexity-Based Alerts (0-3)

| Complexity Score | Alert Count | Category |
|------------------|-------------|----------|
| 1-3 | 0 alerts | Routine |
| 4-7 | 1-2 alerts | Review Required |
| 8-10 | 2-3 alerts | Crisis Level |

### Hierarchical Solutions (1-7 across 4 Tiers)

| Tier | Urgency | Timeline |
|------|---------|----------|
| 1 | IMMEDIATE | < 1 hour |
| 2 | URGENT | 1-2 days |
| 3 | STANDARD | 3-7 days |
| 4 | PLANNED | 3+ weeks |

Each solution includes:
- Step-by-step instructions
- Responsible parties
- Duration estimates
- Success probability
- Source attribution (private data vs external intelligence)

---

## Authentication System

### Multi-Tenant Architecture

- **Client Registry**: Master database stores all clients
- **Subdomain Routing**: `{client_id}.vigilsecure.com`
- **Isolated Data**: Each client has dedicated database connection
- **Custom Domains**: Per-client email domains

### User Hierarchy

```
Coresight Group (Platform Owner)
    └── Client Organizations
            ├── Admin (1 per client)
            │   ├── Create users
            │   ├── Manage access
            │   └── View activity logs
            └── Users (multiple per client)
                └── Risk analysis access
```

### Security Features

- Password hashing (PBKDF2 with salt)
- Session expiration (configurable)
- Account lockout (after failed attempts)
- 6-digit email verification codes
- Rate limiting on login attempts
- Complete activity audit trail

---

## Database Schemas

All databases use PostgreSQL via Supabase.

### Master Database (`supabase-schema-master.sql`)

**Platform Administration Tables:**
- `clients` - Client organizations registry
- `admins` - Admin accounts (one per client)
- `users` - Regular user accounts
- `sessions` - Active login sessions
- `auth_codes` - Verification codes
- `activity_logs` - Audit trail
- `data_sources` - Connected data sources
- `sync_jobs` - ETL job tracking
- `api_keys` - Client API credentials
- `webhooks` - Outbound notifications
- `usage_metrics` - Billing/usage tracking

### Client Database (`supabase-schema-client.sql`)

**Per-Client Risk Data Tables:**
- `risks` - Risk records with 768-dim vectors
- `documents` - Uploaded documents
- `document_chunks` - Chunked document segments
- `suppliers` - Supplier database
- `inventory_items` - Inventory management
- `contracts` - Contract tracking
- `procurement_history` - Recommendations

**Features:**
- pgvector extension for semantic search
- JSONB for flexible metadata
- Row-level security (RLS)
- Composite indexes for performance

---

## Quick Start

### Prerequisites

- Python 3.11+
- Supabase account (PostgreSQL with pgvector)
- Docker (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/vigil-ai.git
cd vigil-ai/Model_Code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r Backend/requirements.txt
```

### Configuration

1. Copy environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your credentials:
```
# Required - Supabase Master (Platform Admin)
SUPABASE_MASTER_URL=https://your-master.supabase.co
SUPABASE_MASTER_KEY=your-master-anon-key
SUPABASE_MASTER_SERVICE_KEY=your-master-service-key

# Required - Supabase Client (Risk Data)
SUPABASE_URL=https://your-client.supabase.co
SUPABASE_KEY=your-client-anon-key
SUPABASE_SERVICE_KEY=your-client-service-key

# Required - Grok AI
XAI_API_KEY=xai-xxxxx

# Flask
FLASK_ENV=development
FLASK_SECRET_KEY=your-secret-key

# Optional
GROK_MODEL=grok-3
```

### Running Locally

```bash
cd Backend
python app.py
```

API available at `http://localhost:5000`

### Running with Docker

```bash
cd Docker
docker-compose up --build
```

---

## Production Deployment

### Architecture

```
Internet
    ↓
Nginx (SSL termination, reverse proxy)
    ↓
Gunicorn (WSGI, 4 workers, 2 threads)
    ↓
Flask Application
    ↓
PostgreSQL (Supabase) + MySQL
```

### Hostinger VPS Deployment

1. **Upload code to server:**
```bash
scp -r Model_Code user@your-vps:/var/www/vigil/
```

2. **Upload secrets:**
```bash
scp .env user@your-vps:/var/www/vigil/Keys_Security/env
```

3. **Run setup script:**
```bash
ssh user@your-vps
export VIGIL_DOMAIN=yourdomain.com
export VIGIL_EMAIL=admin@yourdomain.com
cd /var/www/vigil/backend/deployment/hostinger
chmod +x setup.sh && ./setup.sh
```

4. **Start services:**
```bash
sudo systemctl start vigil
sudo systemctl start nginx
```

### SSL Certificates

```bash
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_MASTER_URL` | Master Supabase project URL | Yes |
| `SUPABASE_MASTER_KEY` | Master anon key | Yes |
| `SUPABASE_MASTER_SERVICE_KEY` | Master service role key | Yes |
| `SUPABASE_URL` | Client Supabase project URL | Yes |
| `SUPABASE_KEY` | Client anon key | Yes |
| `SUPABASE_SERVICE_KEY` | Client service role key | Yes |
| `XAI_API_KEY` | Grok AI API key (X.AI) | Yes |
| `FLASK_SECRET_KEY` | Flask session secret | Yes |
| `FLASK_ENV` | development/production | No |
| `GROK_MODEL` | Grok model version (default: grok-3) | No |
| `GUNICORN_WORKERS` | Number of workers | No |
| `VIGIL_DOMAIN` | Production domain | No |

---

## Frontend Features

### Login Portal
- Dual panels (User/Admin)
- Password visibility toggle
- Multi-tenant routing
- First-time admin verification

### Main Interface
- **Chat**: Real-time risk analysis with file attachments
- **Search**: Semantic search across all documents
- **Alerts**: Severity-based alert panel with details
- **History**: Previous analyses and sessions
- **Settings**: Account and admin management

### Admin Features
- User creation and management
- Activity log viewing
- Last login tracking
- Credential revelation (with verification)

---

## Security

- **Data Validation**: Three-tier validation by data type
- **Encryption**: TLS for transit, encrypted secrets at rest
- **Authentication**: JWT + session-based with expiration
- **Authorization**: Role-based access control
- **Audit Trail**: Complete activity logging
- **Rate Limiting**: Protection against abuse
- **CSP Headers**: Strict content security policy
- **HSTS**: HTTP Strict Transport Security enabled

---

## Dependencies

**Core:**
- Flask, Gunicorn, python-dotenv

**Database:**
- Supabase, psycopg2

**AI/ML:**
- transformers, torch, numpy, requests (for Grok API)

**Document Processing:**
- PyPDF2, python-docx, openpyxl, pandas

**Security:**
- cryptography, PyJWT

**HTTP:**
- httpx, requests, aiohttp

---

## License

Apache 2.0

---

## Support

For issues and feature requests, please open an issue on GitHub.

---

**VIGIL: Enterprise Risk Intelligence at Scale**

Protecting organizations with AI-powered risk detection and mitigation.
