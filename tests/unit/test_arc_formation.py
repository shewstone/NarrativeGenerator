from narrative_engine.composition.formation import arc_formation_gaps, arc_formation_status


def test_arc_requires_three_episodes_and_two_phases():
    assert arc_formation_status(3, 2) == "formed"
    assert arc_formation_status(2, 2) == "candidate"
    assert arc_formation_status(3, 1) == "candidate"


def test_candidate_gaps_are_actionable():
    assert arc_formation_gaps(1, 1) == [
        "needs 2 more episode(s)",
        "needs 1 more distinct phase(s)",
    ]
    assert arc_formation_gaps(4, 3) == []
