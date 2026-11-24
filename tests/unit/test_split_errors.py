import numpy as np
import pytest

from edgar_study.split_errors import split_errors


def test_split_errors_reports_fit_and_held_out_error_for_each_cell_group():
    observed = np.zeros((4, 4))
    predicted = np.array(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [0.0, 0.0, 3.0, 3.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    groups = {"discovery": np.array([0, 1]), "locked_test": np.array([2, 3])}

    errors = split_errors(observed, predicted, np.array([0, 1]), np.array([2, 3]), groups)

    assert list(errors) == ["discovery", "locked_test"]
    assert errors["discovery"]["fit"] == pytest.approx(1.0)
    assert errors["discovery"]["held_out"] == pytest.approx(4.0)
    assert errors["locked_test"]["fit"] == pytest.approx(0.0)
    assert errors["locked_test"]["held_out"] == pytest.approx(5.0)
    assert errors["locked_test"]["held_out_sem"] == pytest.approx(4.0)
    assert errors["discovery"]["cells"] == 2


def test_split_errors_rejects_overlapping_trial_halves():
    with pytest.raises(ValueError, match="overlap"):
        split_errors(np.zeros((1, 3)), np.zeros((1, 3)), np.array([0, 1]), np.array([1, 2]), {})
