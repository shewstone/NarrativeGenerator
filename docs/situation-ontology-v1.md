# Situation Ontology v1

The engine's primary reading is scale-neutral. Finance, politics, psychology,
religion, ecology, and culture are domains in which change appears; none is the
default model of change.

## Situation record

Each episode can carry:

- a focal scope: person, dyad, group, faction, idea, discourse, movement,
  organization, party, institution, polity, civilization, region, or system;
- a parent scope, so a faction can sit inside a party, a party inside a polity,
  and a polity inside a civilization;
- one analytical scale and multiple domain facets;
- six continuous axes from `-1` to `+1`: capacity, cohesion, pressure,
  legitimacy, adaptability, and agency;
- one supported change pattern and one or more broad mechanism families;
- optional detailed mechanism tags and legacy arc/phase labels.

Unknown focal scopes retain their source wording, evidence quote, boundary
note, and confidence. Exact, high-confidence aliases resolve to the versioned
scope registry. Weak claims remain visible but cannot become hard composition
partitions.

## Change patterns

1. emergence and gathering
2. expansion and consolidation
3. saturation and overreach
4. tension and contestation
5. fragmentation and release
6. retreat and preservation
7. turning and reorientation
8. renewal and integration
9. succession and transfer

These name a mode of change, not a moral judgment. “Rise” must be grounded in
observable proxies such as adoption, membership, territorial reach,
institutional presence, resource capacity, policy uptake, audience share, or
influence. A source's assertion that something is rising is evidence of that
source's framing, not automatically evidence that the rise occurred.

## Subgroups and contested categories

The Communist Party of China can be represented as a party nested in China,
with episodes tracing emergence, expansion, contestation, consolidation,
adaptation, and succession. A faction within the party can be a still more
local focal scope with its own trajectory.

A label such as “wokeness” requires different handling. Depending on the
source and evidence, it may refer to an idea, discourse, loose movement,
coalition, or a set of institutional practices. Extraction preserves the
source's term and records boundary ambiguity instead of manufacturing a
single membership organization. Conflicting sources can therefore support
competing reviewable scope claims.

## Multi-scale reading

An episode's focal scope identifies the most local subject whose trajectory
changes. Existing episode-to-cycle memberships can attach that same episode to
additional parent-scale readings. In the graph, a scope filter includes nested
scope paths, while a non-causal chronological edge shows observations of the
same focal subject over time.

Version 1 uses one stable containment parent per registered scope plus a raw
parent claim for newly observed scopes. Overlapping coalitions, changing party
affiliations, and territorial membership that changes through time will need a
future temporal scope-relation table; they should not be forced into the
single-parent field.
