# Narrative Engine

A system for qualitative historical forecasting through narrative arc extraction.

## Overview

The Narrative Engine treats history as a corpus of stories with recurring arcs. It uses LLMs to extract, classify, and structure those arcs at multiple time scales, then generates forecasts ("theses") by asking: **"Given where we are in the story, how do stories like this usually end?"**

## Core Concepts

- **Episode**: Atomic narrative unit (e.g., "Weimar hyperinflation, 1921–1923")
- **Arc**: Archetypal shape (e.g., "credit boom and bust", "hubris–nemesis")
- **Cycle**: Fractal containers (civilizational → institutional → generational)
- **Thesis**: Generated forecast with historical analogs and confidence intervals

## Architecture

```
src/narrative_engine/
├── ingestion/        # Text ingestion from books, articles, archives
├── extraction/       # LLM pipeline: segmentation, extraction, classification,
│                     # linking; controlled role vocabulary (roles.py)
├── storage/          # Database: SQLAlchemy models, pgvector, repositories
├── retrieval/        # Analog retrieval, dual embeddings + epoch versioning,
│                     # re-embed batch job (reembed.py)
├── composition/      # Arc-instance composition: identity resolution,
│                     # scope-partitioned merging (design doc Sec 6.2 stage 6)
├── taxonomy/         # Discovery/promotion scaffolding, role-residue metrics
├── thesis/           # Thesis synthesis from historical analogs
├── evaluation/       # Masked-ending harness, baselines, Brier/calibration
├── scopes.py         # Versioned scope registry + alias resolver
└── data/             # Versioned data artifacts (scope_registry.json)
```

## Quick Start

### Docker (recommended — this is what CI runs)

```bash
docker compose up -d db
# The test suite uses a separate database that compose does not auto-create:
docker compose exec db psql -U postgres -c "CREATE DATABASE narrative_engine_test"

docker compose run --rm app                     # full test suite
docker compose run --rm -v ./alembic:/app/alembic app alembic upgrade head
```

### Always-on server: drop directory + dashboard

```bash
cp .env.example .env
# Add your Venice key to NE_LLM_API_KEY in .env
docker compose up -d server     # dashboard on http://localhost:8000
```

Drop books/articles into `data/raw/` — the watcher picks them up
automatically (settled files only, so half-copied files are safe), guards
against duplicates by content hash (same bytes under any filename become a
visible `duplicate` row, never reprocessed), chunks them, and — when
`NE_LLM_API_KEY` is set — extracts episodes, embeds
them, and composes trajectories. The primary ontology is scale-neutral:
episodes receive a change pattern, personal-to-system scale, domain facets,
six situation axes, mechanism families, and a hierarchical focal scope.
That scope can be a party, faction, movement, idea, or discourse nested in a
larger polity or civilization. The dashboard colors the graph by change
pattern and filters it by scale, domain, and any parent/child scope; legacy
arc labels remain available for compatibility. **No auth: localhost/dev use
only.**

See [Situation Ontology v1](docs/situation-ontology-v1.md) for the change
patterns, subgroup treatment, evidence rules, and current hierarchy limits.

The Docker configuration uses Venice's OpenAI-compatible API with
`zai-org-glm-4.7-flash` for segmentation, `zai-org-glm-5-1` for structured
extraction, and `deepseek-v3.2` for classification and linking. Override any
stage with `NE_SEG_MODEL`, `NE_EXTRACT_MODEL`, `NE_CLASSIFY_MODEL`, or
`NE_LINK_MODEL`. Pairwise linking always checks adjacent episodes; distant
pairs must pass a cheap entity, place, time, or lexical prefilter. Its defaults
can be tuned with `NE_LINK_NEIGHBOR_WINDOW`, `NE_LINK_MAX_YEAR_GAP`, and
`NE_LINK_MIN_LEXICAL_OVERLAP`. `NE_TAU_SCOPE` (default `0.6`) prevents weak
scope claims from becoming hard composition partitions; `NE_TAU_CLASS` and
`NE_TAU_ROLE` apply the same no-forced-choice discipline to patterns and
actor roles.

### Local installation

```bash
pip install -e ".[dev]"
createdb narrative_engine
alembic upgrade head
pytest -v
```

### Fixture gates (run before changing embeddings, the render, or composition)

```bash
make fixture-gates    # analog fixture (Sec 6.3) + composition fixture (Sec 6.6)
make tune-thresholds  # justify per-scale temporal thresholds against the fixture
make reembed          # bring stale-epoch embeddings to the current epoch
```

The analog gate's baseline (2026-08-28, render-v0.9.0 + all-MiniLM-L6-v2):
pair_recall@5 = 0.900, MRR = 0.753 over 30 cross-era pairs + 30 distractors.
The floor (0.85) is a ratchet: raise it when the render improves, never lower
it to make a change pass.

### Example Usage

```python
from narrative_engine.models import Episode, ArcType, ArcPhase
from narrative_engine.storage.database import db_manager
from narrative_engine.storage.repositories import RepositoryFactory

async with db_manager.session() as session:
    factory = RepositoryFactory(session)
    
    # Create episode
    episode = await factory.episodes.create(
        Episode(
            title="1929 Stock Market Crash",
            summary="The collapse of the US stock market in October 1929",
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.PANIC,
        )
    )
    
    # Retrieve by arc type
    similar = await factory.episodes.get_by_arc_type(
        ArcType.CREDIT_BOOM_AND_BUST.value
    )
```

## How To Use

### Data Ingestion

The Narrative Engine processes historical texts to extract structured narrative episodes. Here's how to add your own data:

#### Supported Data Types

| Type | Format | Examples |
|------|--------|----------|
| **Books** | PDF, EPUB, TXT | Kindleberger's *Manias, Panics, and Crashes*, Taleb's *The Black Swan* |
| **Articles** | TXT, Markdown | Academic papers, journalism, historical essays |
| **Archives** | JSON, CSV | Structured datasets like Reinhart-Rogoff crisis database |
| **Timelines** | JSON | Encyclopedic chronologies for scaffolding |

#### Ingestion Process

1. **Place raw files** in the `data/raw/` directory:
   ```
   data/raw/
   ├── books/kindleberger-mania-2026.pdf
   ├── articles/taleb-black-swan.txt
   └── archives/reinhart-rogoff-crisis.json
   ```

2. **Run the ingestion pipeline**:
   ```bash
   python -m narrative_engine.ingestion \
     --source-dir data/raw/ \
     --output-dir data/processed/
   ```

3. **Processed output** will be:
   - Text chunks (~2-8k tokens, respecting chapter boundaries)
   - Metadata (author, publication date, historiographic school)
   - Chunk IDs for provenance tracking

### Extraction Pipeline

After ingestion, run the LLM extraction pipeline:

```python
from narrative_engine.extraction.pipeline import ExtractionOrchestrator

orchestrator = ExtractionOrchestrator()

# Process a single chunk
episodes = await orchestrator.process_text(
    text="The stock market crash of 1929...",
    source_id="kindleberger-1929",
)

# Process entire directory
results = await orchestrator.process_directory("data/processed/")
```

**Pipeline stages**:
1. **Segmentation** → Episode boundaries + one-line summaries
2. **Extraction** → Actors, conditions, mechanics, resolution, consequences
3. **Classification** → Arc type + phase + confidence
4. **Linking** → Causal connections, cycle membership

### Expected Output

For each processed episode, you'll get:

```python
Episode(
    title="1929 Stock Market Crash",
    summary="The collapse of the US stock market in October 1929",
    arc_type=ArcType.CREDIT_BOOM_AND_BUST,
    arc_phase=ArcPhase.PANIC,
    actors=[
        Actor(name="Herbert Hoover", role="President"),
        Actor(name="J.P. Morgan Jr.", role="Banker"),
    ],
    initiating_conditions=[
        "Speculative excess",
        "Margin trading widespread",
    ],
    escalation_mechanics=[
        "Panic selling cascades",
        "Margin calls force liquidation",
    ],
    resolution="Market bottomed in 1932, down 89% from peak",
    consequences=[
        "Great Depression begins",
        "Banking crisis follows",
        "New Deal reforms enacted",
    ],
    # Dual embeddings, never swapped (design doc Sec 3.3a): surface = identity
    # ("same happening?"), structural = analogy ("same shape?"). Each carries
    # the (render, model) epoch that produced it; retrieval only compares
    # vectors from the current epoch.
    surface_embedding=[...],      # 384-dim, epoch-stamped
    structural_embedding=[...],   # 384-dim, epoch-stamped
)
```

### Generating Forecasts

Once episodes are extracted, generate forecasts ("theses"):

```python
from narrative_engine.retrieval.embeddings import EmbeddingGenerator
from narrative_engine.retrieval.analog_retrieval import AnalogRetriever
from narrative_engine.thesis.generator import ThesisGenerator

# Embed your current situation
query = Episode(
    title="2024 Market Conditions",
    summary="High valuations, AI bubble concerns, rising rates",
    arc_type=ArcType.CREDIT_BOOM_AND_BUST,
    arc_phase=ArcPhase.DISTRESS,
)

embedder = EmbeddingGenerator()
retriever = AnalogRetriever()
generator = ThesisGenerator()

# Retrieve historical analogs
query_embedding = embedder.generate_for_episode(query)
analogs = await retriever.find_analogs(query_embedding, query)

# Generate thesis
thesis = generator.generate(query, analogs)

print(f"Dominant continuation: {thesis.dominant_continuation}")
print(f"Confidence: {thesis.confidence}")
print(f"Watch conditions: {thesis.watch_conditions}")
```

**Example output**:

```
Dominant continuation: Soft landing likely (65% probability)
Confidence: 0.72
Watch conditions:
  - Credit spreads widening
  - Yield curve inversion persistence
  - Fed policy shift signals

Based on 12 historical analogs:
  - 1929 Crash (phase match: 0.85)
  - 2008 Financial Crisis (phase match: 0.78)
  - 1987 Black Monday (phase match: 0.72)
```

### Evaluation & Backtesting

The masked-ending harness (design doc Sec 6.6) runs the whole loop —
snapshot the corpus at a cutoff year with outcomes masked at the data
layer, retrieve analogs, generate theses, score against the known
continuations, and compare against the persistence baseline:

```python
from datetime import datetime, timezone
from narrative_engine.evaluation.harness import run_backtest

report = await run_backtest(
    session,
    corpus,                                    # unmasked; harness masks it
    cutoff=datetime(1930, 1, 1, tzinfo=timezone.utc),
)
print(report.summary())
# {"cases": ..., "mean_thesis_brier": ..., "mean_persistence_brier": ...,
#  "skill_vs_persistence": ...}   # <=0 means the machinery isn't paying rent
```

Leakage canaries are hard errors inside the harness: post-cutoff episodes
never enter the snapshot database, and any analog carrying an outcome not
knowable at the cutoff raises `LeakageError`. Brier scores use the 0–1
single-probability convention (0 = perfect, 1 = worst).

## Development

### Project Structure

- **Phase 1** ✅: Core models (Episode, Arc, Cycle, Thesis)
- **Phase 2** ✅: Database layer (SQLAlchemy, pgvector, ORM)
- **Phase 3** ✅: Repository pattern (CRUD, semantic search)
- **Phase 4** ✅: LLM extraction pipeline
- **Phase 5** ✅: Analog retrieval and thesis generation
- **Phase 6** ✅: Evaluation and backtesting

### PR History

| PR | Description | Status |
|----|-------------|--------|
| #2-#7 | Core → Database → Repository → Extraction | ✅ Merged |
| #8 | Vector Embeddings + Analog Retrieval | ✅ Merged |
| #9 | Thesis Generation | ✅ Merged |
| #10 | Testing + Documentation | ✅ Merged |
| #11 | Missing models fix | ✅ Merged |

## Design Principles

1. **Fractal time**: Cycles nest (civilizational → institutional → generational)
2. **Narrative arcs**: Episodes instantiate archetypal patterns
3. **Phase completion**: Forecasting by locating current position in arc
4. **Provenance**: Every claim traceable to source passages
5. **Evaluation**: Rigorous backtesting with Brier score calibration

## Related Documentation

- [Narrative Finance: Qualitative Forecasting](https://github.com/shewstone/NarrativeGenerator/wiki/Narrative-Finance-Qualitative-Forecasting)
- [Narrative Engine Technical Design](https://github.com/shewstone/NarrativeGenerator/wiki/Narrative-Engine-Technical-Design)
- [Fractal Time Framework](https://github.com/shewstone/NarrativeGenerator/wiki/fractal-time)

## License

MIT
