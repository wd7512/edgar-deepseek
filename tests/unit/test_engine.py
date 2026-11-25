from argparse import Namespace

from edgar_study.engine import IMAGE_READER_PROMPT, split_model


def test_split_model_carries_a_tier_effort():
    assert split_model("m#high") == ("m", "high")
    assert split_model("m") == ("m", None)


def test_large_model_defaults_to_same_model_with_high_effort():
    from edgar_study.engine import large_model

    assert large_model(Namespace(model="m", large_model=None, large_reasoning_effort="high")) == "m#high"
    assert large_model(Namespace(model="m", large_model="big", large_reasoning_effort="high")) == "big"


def test_image_reader_names_only_the_shape_features_upstream_prompts_already_list():
    for hint in ("cusp", "power", "exponent", "generalized", "kink"):
        assert hint not in IMAGE_READER_PROMPT.lower()
