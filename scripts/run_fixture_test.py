"""Run composition fixture validation against the real composition pipeline.

Human-readable CLI wrapper around the same fixture + algorithm exercised
by tests/unit/test_composition/test_fixture.py. Delegates to
compose_arc_instances_from_episodes (the actual Sec 6.2 stage 6 staged
pipeline) rather than a parallel reimplementation, so this script's output
reflects what the real pipeline does.
"""

import asyncio
import sys

sys.path.insert(0, 'tests/fixtures')

from composition_fixture import validate_composition_pipeline

from narrative_engine.composition import compose_arc_instances_from_episodes


async def real_pipeline(episodes):
    """Adapter: fixture cases each use a single arc_type."""
    if not episodes:
        return []
    return compose_arc_instances_from_episodes(episodes, arc_type=episodes[0].arc_type)


async def main():
    results = await validate_composition_pipeline(real_pipeline, verbose=True)

    print()
    print('='*70)
    print('COMPOSITION FIXTURE RESULTS')
    print('='*70)
    print(f'Positive (should merge):     {results["positive_passed"]}/{results["positive_total"]}')
    print(f'Negative (should NOT merge): {results["negative_passed"]}/{results["negative_total"]}')
    print(f'Edge cases:                  {results.get("edge_passed", 0)}/{results.get("edge_total", 0)}')
    print()

    if results['recommendations']:
        print('Threshold Tuning Recommendations:')
        for rec in results['recommendations']:
            print(f'  • {rec}')
        print()
        print('Status: ⚠️  FIXTURE NOT PASSING - tune thresholds')
    else:
        print('Status: ✅ FIXTURE PASSING - thresholds validated!')

    # Overall pass/fail
    total_pass = results['positive_passed'] + results['negative_passed']
    total = results['positive_total'] + results['negative_total']
    if total:
        print(f'Overall: {total_pass}/{total} ({100*total_pass//total}% pass rate)')
    else:
        print('Overall: no fixture cases found')


if __name__ == "__main__":
    asyncio.run(main())
