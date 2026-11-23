import numpy as np

from edgar_study.datasets import prepare, synthetic_recording


def test_strict_holdout_groups_are_disjoint_and_cover_eligible_cells():
    x, y, groups, indices = prepare(*synthetic_recording(), "strict-holdout-v1")
    parts = [set(groups[name].tolist()) for name in ("discovery", "selection", "locked_test")]
    assert sum(len(p) for p in parts) == x.shape[0] == y.shape[0]
    assert parts[0].isdisjoint(parts[1]) and parts[0].isdisjoint(parts[2]) and parts[1].isdisjoint(parts[2])
    assert len(indices["fit_trial_columns"]) + len(indices["score_trial_columns"]) == x.shape[1]


def test_split_is_deterministic():
    first = prepare(*synthetic_recording(), "strict-holdout-v1")[3]
    second = prepare(*synthetic_recording(), "strict-holdout-v1")[3]
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_responses_are_scaled_to_norm_100():
    _, y, _, _ = prepare(*synthetic_recording(), "paper-fidelity")
    assert np.allclose(np.linalg.norm(np.asarray(y), axis=1), 100)
