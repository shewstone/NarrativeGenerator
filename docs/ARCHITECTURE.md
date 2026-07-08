# Narrative Engine Architecture

## System Overview

The Narrative Engine is a system for qualitative historical forecasting through narrative arc extraction. It treats history as a corpus of stories with recurring patterns and uses LLMs to extract, classify, and structure those arcs for prediction.

## Core Concepts

### Episode
An **Episode** is the atomic narrative unit — a bounded stretch of historical action with:
- **Temporal bounds**: Start/end dates with precision level
- **Setting**: Location and context
- **Actors**: Participants with roles
- **Arc classification**: Type (e.g., "credit boom and bust") and phase (e.g., "panic")
- **Provenance**: Source passages for every claim

### Arc
An **Arc** is an archetypal shape that episodes instantiate:
- Credit boom and bust (Minsky-Kindleberger)
- Hubris–nemesis (tragedy pattern)
- Rise and overextension
- Decadence and renewal

Each arc has defined phases (e.g., boom → euphoria → distress → panic → revulsion).

### Cycle
A **Cycle** is a recursive container providing fractal structure:
- Civilizational (~centuries)
- Institutional (~decades)
- Generational (~20-25 years)
- Episodic (individual events)

### Thesis
A **Thesis** is a generated forecast:
- Query: Present-day situation
- Analogs: Retrieved historical episodes
- Dominant continuation: Most likely outcome
- Alternatives: Other plausible outcomes with base rates
- Watch-for indicators: Conditions that distinguish branches

## Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Text   │────▶│  Extraction  │────▶│  Episodes   │
│  (books,    │     │  Pipeline    │     │  (DB)       │
│  articles)  │     │  (LLM)       │     │             │
└─────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │  Cycle       │
                                        │  Assignment  │
                                        └──────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Forecast   │◀────│  Thesis      │◀────│  Analog      │
│  (output)   │     │  Generation  │     │  Retrieval   │
└─────────────┘     └──────────────┘     └──────────────┘
```

## Module Structure

### ingestion/
**Purpose**: Read and normalize source documents

**Key operations**:
- Format normalization (EPUB, PDF, Markdown)
- OCR for image-based PDFs
- Structural parsing (preserve chapters/sections)
- Narrative-aware chunking (~2-8k tokens)

**Output**: Normalized chunks with metadata (work ID, author, historiographic school)

### extraction/
**Purpose**: LLM pipeline for structured extraction

**Stages**:
1. **Segmentation**: Identify episode boundaries
2. **Extraction**: Pull actors, conditions, mechanics, resolution
3. **Classification**: Assign arc type and phase with confidence
4. **Linking**: Entity resolution across sources, causal connections

**Versioning**: All prompts and schemas versioned for reproducibility

### storage/
**Purpose**: Database persistence and retrieval

**Components**:
- **database.py**: Connection pooling, session management
- **orm_models.py**: SQLAlchemy ORM models
- **repositories.py**: Repository pattern for CRUD operations
- **config.py**: Environment-based configuration

**Technology Stack**:
- PostgreSQL 16 with pgvector extension
- Async SQLAlchemy 2.0
- 768-dimensional embeddings (sentence-transformers)
- IVFFlat index for approximate nearest neighbor search

### retrieval/
**Purpose**: Find historical analogs for present situations

**Methods**:
- **Semantic search**: Vector similarity over episode embeddings
- **Graph traversal**: Follow causal links, cycle membership
- **Arc matching**: Same arc type, similar phase position
- **Scale filtering**: Episode behavior varies by cycle context

### generation/
**Purpose**: Synthesize theses from retrieved analogs

**Process**:
1. Retrieve k nearest analogs
2. Analyze phase transitions: from phase N, what happened?
3. Synthesize: dominant outcome, alternatives, watch-fors
4. Cite: Link every claim to source episodes

### evaluation/
**Purpose**: Measure and calibrate forecasting accuracy

**Metrics**:
- **Brier score**: Probabilistic forecast calibration
- **Base rate comparison**: vs. naive persistence
- **Leakage control**: Ensure no train/test contamination

## Database Schema

### episodes
Primary table for narrative units.

```sql
CREATE TABLE episodes (
    id UUID PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    summary TEXT NOT NULL,
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    arc_type VARCHAR(50),  -- ArcType enum
    arc_phase VARCHAR(50), -- ArcPhase enum
    phase_confidence FLOAT DEFAULT 0.0,
    embedding VECTOR(768), -- pgvector
    -- JSON fields for flexible metadata
    initiating_conditions JSONB DEFAULT '[]',
    escalation_mechanics JSONB DEFAULT '[]',
    consequences JSONB DEFAULT '[]',
    secondary_arcs JSONB DEFAULT '[]',
    extracted_from JSONB DEFAULT '[]',
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER DEFAULT 1
);
```

### cycles
Fractal cycle hierarchy.

```sql
CREATE TABLE cycles (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    scale VARCHAR(50) NOT NULL,  -- CycleScale enum
    description TEXT,
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    parent_cycle_id UUID REFERENCES cycles(id),
    dominant_arc_types JSONB DEFAULT '[]',
    phase_estimate VARCHAR(50),
    framework_source VARCHAR(255)
);
```

### theses
Generated forecasts with evaluation tracking.

```sql
CREATE TABLE theses (
    id UUID PRIMARY KEY,
    query TEXT NOT NULL,
    query_date TIMESTAMP WITH TIME ZONE NOT NULL,
    analog_episode_ids JSONB DEFAULT '[]',
    analog_similarity_scores JSONB DEFAULT '[]',
    dominant_continuation TEXT NOT NULL,
    alternative_continuations JSONB DEFAULT '[]',
    watch_for_indicators JSONB DEFAULT '[]',
    cited_episodes JSONB DEFAULT '{}',
    -- Evaluation
    resolved BOOLEAN DEFAULT FALSE,
    resolution_date TIMESTAMP WITH TIME ZONE,
    resolution_outcome VARCHAR(50),
    brier_score FLOAT,
    -- Metadata
    model_version VARCHAR(100) NOT NULL,
    taxonomy_version VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Design Decisions

### Why Pydantic v2?
- Type safety at runtime and static analysis
- Automatic validation and serialization
- Clear JSON Schema generation
- Integration with FastAPI

### Why Async SQLAlchemy?
- Non-blocking database operations
- Better concurrency for I/O-bound workloads
- Natural fit for FastAPI async endpoints
- Compatible with asyncpg driver

### Why pgvector?
- Native PostgreSQL extension (no external service)
- Supports multiple distance metrics (cosine, L2, inner product)
- IVFFlat and HNSW indexes for approximate search
- ACID compliance with vector operations

### Why Repository Pattern?
- Separation of data access from business logic
- Testability via mock repositories
- Flexibility to change storage backend
- Transaction boundary control

## Scaling Considerations

### Current Limits (v1)
- ~10³–10⁴ books/documents
- Vector search: IVFFlat index suitable for <1M vectors
- Synchronous LLM calls (batch processing)

### Future Scaling
- **10⁵+ documents**: Migrate to pgvector HNSW index, connection pooling
- **10⁶+ documents**: External vector database (Pinecone, Weaviate)
- **Real-time**: Streaming ingestion with Kafka/message queues
- **Distributed**: Celery/Ray for parallel extraction

## Security Considerations

- Database credentials via environment variables
- No secrets in code or repositories
- Prepared statements via SQLAlchemy (SQL injection prevention)
- Input validation via Pydantic models
- Audit trail via extraction_records table

## Related Documents

- [Narrative Finance: Qualitative Forecasting](../wiki/Narrative-Finance-Qualitative-Forecasting.md)
- [Narrative Engine Technical Design](../wiki/Narrative-Engine-Technical-Design.md)
- [API Documentation](API.md) (to be written)
- [Deployment Guide](DEPLOYMENT.md) (to be written)
