# Narrative Engine UI audit

Date: 2026-08-29

## Verdict

The first-load Arc Space is overcrowded: 500 nodes and 745 links form a dense knot that does not communicate arc formation. Filtering to a single scope makes the graph usable, while the chronological card sequence is the clearest expression of the product's core idea and should be the primary view.

## Steps reviewed

1. First load — poor: dense graph, no obvious starting point.
2. Filter to China — improved: 11 episodes and 22 links become inspectable.
3. Select an episode — good evidence, excessive detail without progressive disclosure.
4. Read chronological progression — strongest view, but buried below the 3D plot.
5. Open Operations — useful phase coverage, but an unfiltered wall of 1,091 arc instances.

## Highest-impact changes

1. Start with a scope/subject chooser or a curated example instead of all 500 episodes.
2. Promote the chronological arc/swimlane view above the PCA graph.
3. Replace the 327-option scope dropdown with hierarchical search over canonical scopes.
4. Treat the 3D PCA as an optional similarity-space view.
5. Add search, filtering, grouping, and pagination to Operations.
6. Expose graph nodes and timeline cards as keyboard-accessible controls and provide a list alternative.

## Evidence limits

This review covered the desktop dashboard at 1280×720. It did not include mobile breakpoints, formal contrast measurement, a complete keyboard traversal, or screen-reader testing.
