# Design QA

final result: passed

## Comparison target

- State: China scope, dark theme, desktop dashboard, selected historical episode.
- CSS viewport: 1440 × 1024.
- Device scale factor: 1.
- Implementation browser capture: 1440 × 1024 pixels.
- Source ImageGen captures: 1487 × 1058 pixels, normalized to 1440 × 1024 with proportional resampling before comparison.
- Browser-rendered page dimensions: 1440 × 1024 with no page-level horizontal or vertical overflow in the final River state.

### Source visual truth

- Facet River: `tmp/ui-reference-river-1440x1024.png`
- Arc Storyboard: `tmp/ui-reference-storyboard-1440x1024.png`
- Constellation: `tmp/ui-reference-constellation-1440x1024.png`

### Implementation evidence

- Facet River: `tmp/ui-build-2026-08-29-river.png`
- Arc Storyboard: `tmp/ui-build-2026-08-29-storyboard.png`
- Constellation: `tmp/ui-build-2026-08-29-constellation.png`
- Similarity Space: `tmp/ui-build-2026-08-29-similarity.png`
- Operations: `tmp/ui-build-2026-08-29-operations.png`

## Full-view comparison evidence

All three normalized reference/implementation pairs were opened together at original detail. The implementation preserves the selected dark research language, compact header, cool-gray Inter typography, cyan/blue/purple/amber phase accents, fine panel borders, five-facet hierarchy, shared filtering, selected-event emphasis, and evidence-first inspector.

The River intentionally uses evidence-density bands plus observed subject trajectories instead of copying the reference's invented continuous historical intervals. This is a product/data constraint rather than drift: thickness now communicates the density of real extracted evidence, while colored episode markers communicate phase and hover reveals exact subjects.

The Storyboard matches the six phase columns and five facet rows while using actual corpus episodes. The Constellation matches the enclosure, subgroup, synchronized-timeline, and inspector composition while limiting visible trajectories to available evidence.

Focused crops were not required: both sides were compared at the same 1440 × 1024 dimensions and original-detail rendering, where navigation, controls, matrix cards, plot labels, phase colors, and inspector typography were readable. Exact control state and text were additionally checked through the browser-rendered DOM.

## Required fidelity surfaces

- Fonts and typography: Inter is used at the source-like weights and compact sizes. Headings, eyebrows, labels, card titles, and inspector copy retain clear hierarchy and do not clip.
- Spacing and layout rhythm: 12–20 px section spacing, 7–16 px component padding, 7–11 px radii, and consistent borders match the reference density. The Storyboard now shows all five facet rows and the start of its evidence drawer at 1440 × 1024.
- Colors and visual tokens: near-black background, charcoal panels, cool-gray text, blue selection, cyan relationships, and semantic phase colors are consistent across tabs.
- Image quality and asset fidelity: the designs contain no photographic or illustrative raster assets. Plotly renders the data visualizations, Remix Icon supplies UI icons, and no placeholder or handcrafted icon assets remain.
- Copy and content: all visible episode, scope, date, confidence, phase, mechanism, source, and coverage content comes from the running corpus. The UI avoids invented financial KPIs or fabricated historical records.

## Interaction and browser verification

- Opened all five tabs: Facet River, Arc Storyboard, Constellation, Similarity Space, and Operations.
- Verified shared China filtering returned 66 episodes from the complete 1,755-episode corpus.
- Verified Compare adds United States and changes the matching total from 66 to 98, then cleanly disables.
- Selected a Storyboard episode and confirmed the shared inspector updated to the same episode.
- Toggled a Constellation facet off and back on.
- Verified Similarity Space loads its capped PCA data only when opened and renders 11 China-filtered projected episodes.
- Verified Operations renders 30 paginated arcs, 16 recent documents, and 20 pending claims rather than the former unbounded wall.
- Checked browser console warnings and errors after the interaction pass: none.
- Checked responsive layout at 900 × 800: viewport width and document width were both 900 px; navigation and controls remained visible.
- Automated test suite: 394 passed.

## Comparison history

### Iteration 1

- [P1] River did not communicate arc formation strongly enough because real subject paths were sparse.
  - Fix: added non-fictional evidence-density bands behind the observed subject trajectories.
  - Post-fix evidence: `tmp/ui-build-2026-08-29-river.png`.
- [P2] Storyboard rows were too tall to scan at the target viewport.
  - Fix: reduced row rhythm and prioritized one selected/high-confidence episode per cell with an explicit overflow count.
  - Post-fix evidence: `tmp/ui-build-2026-08-29-storyboard.png`.
- [P1] Cross-facet relation rules were too broad and recreated visual noise.
  - Fix: constrained inferred context by shared scope, time distance, exact subject/container, and selected arc; the Constellation now shows only evidence-relevant links.
  - Post-fix evidence: `tmp/ui-build-2026-08-29-constellation.png`.

### Final pass

No actionable P0, P1, or P2 findings remain.

## Follow-up polish

- [P3] A searchable hierarchical scope picker would be faster than the current native select for very large corpora.
- [P3] Future ingestion can improve River label density in historically sparse facets without changing this layout.
