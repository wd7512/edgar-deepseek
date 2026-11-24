import numpy as np
import pytest

from edgar_study.delta import delta_analysis


def test_delta_analysis_summarises_gain_and_size_of_the_winner_over_the_paper():
    paper_r2 = np.array([0.30, 0.20, 0.50, 0.40])
    winner_r2 = np.array([0.31, 0.25, 0.49, 0.40])
    paper_curves = np.array([[0.0, 2.0, 0.0], [1.0, 3.0, 1.0], [0.0, 4.0, 0.0], [0.0, 1.0, 0.0]])
    winner_curves = paper_curves + np.array(
        [[0.0, 0.1, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    )

    result = delta_analysis(paper_r2, winner_r2, paper_curves, winner_curves, n_examples=2)

    assert result["gain"]["median"] == pytest.approx(0.005)
    assert result["gain"]["share_better"] == pytest.approx(0.5)
    assert result["gain"]["share_worse"] == pytest.approx(0.25)
    assert sum(result["gain"]["hist"]) == 4
    assert result["size"]["median"] == pytest.approx(0.275)
    assert result["size"]["share_over_10pct"] == pytest.approx(0.5)
    assert sum(result["size"]["hist"]) == 4
    assert result["rows"] == [1, 0]


def test_delta_analysis_is_zero_when_the_winner_is_the_paper():
    curves = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0]])
    r2 = np.array([0.3, 0.4])

    result = delta_analysis(r2, r2, curves, curves.copy())

    assert result["gain"]["share_better"] == 0
    assert result["gain"]["share_worse"] == 0
    assert result["size"]["median"] == 0
    assert result["size"]["share_over_10pct"] == 0
