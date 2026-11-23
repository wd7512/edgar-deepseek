from argparse import Namespace

from edgar_study.engine import split_model


def test_split_model_carries_a_tier_effort():
    assert split_model("m#high") == ("m", "high")
    assert split_model("m") == ("m", None)


def test_large_model_defaults_to_same_model_with_high_effort():
    from edgar_study.engine import large_model

    assert large_model(Namespace(model="m", large_model=None, large_reasoning_effort="high")) == "m#high"
    assert large_model(Namespace(model="m", large_model="big", large_reasoning_effort="high")) == "big"
