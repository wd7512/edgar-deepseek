from edgar_study.frontier import grammar
from edgar_study.frontier.ideas import Idea
from edgar_study.frontier.prompts import (
    EXAMPLE,
    PROMPTS,
    generated_prompt,
    generated_prompt_v2,
    idea_prompt,
    tool_note,
)


def test_revisit_prompt_lists_tried_equations_without_scores():
    node = Idea(
        "idea-1", "revisit", "width", "Split the widths.", (("double_gaussian",),), "frontier", "Widths"
    )
    prompt = generated_prompt(
        {"node": "idea-1", "kind": "revisit", "mode": "revisit", "parents": ["double_gaussian"]},
        {"idea-1": node},
        {},
        4,
        ("b + a*cos(theta)",),
    )
    assert "do not repeat" in prompt
    assert "b + a*cos(theta)" in prompt
    assert "score" not in prompt.lower()


SCORE_WORDS = ("mse", "score", "loss", "ranking", "best")


def _prompts():
    node = Idea(
        "idea-1", "single_idea", "width", "Split the widths.", (("double_gaussian",),), "frontier", "W"
    )
    generated = generated_prompt(
        {"node": "new", "kind": "new", "mode": "single_idea", "parents": ["double_gaussian"]}, {}, {}, 4
    )
    return generated, idea_prompt(node, (), {}, {}, 4)


def test_both_prompts_ask_for_a_single_expression_and_a_complete_nesting_map():
    for prompt in _prompts():
        assert "single expression" in prompt and "no assignments" in prompt
        assert "every parameter" in prompt


def test_the_tool_note_explains_checking_and_how_many_follow_up_turns_there_are():
    note = tool_note(3)
    assert '{"check": [' in note and "3 follow-up turns" in note
    assert "nesting map" in note
    assert all(word not in note.lower() for word in SCORE_WORDS)


def test_with_no_patience_there_is_no_tool_note():
    assert tool_note(0) == ""


SCALE = {"baseline": 0.4, "peak": 12.0}


def _v2_prompts():
    node = Idea(
        "idea-1", "single_idea", "width", "Split the widths.", (("double_gaussian",),), "frontier", "W"
    )
    new = generated_prompt_v2(
        {"node": "idea-2", "kind": "new", "mode": "single_idea", "parents": ["double_gaussian"]},
        {},
        {},
        4,
        scale=SCALE,
    )
    revisit = generated_prompt_v2(
        {"node": "idea-1", "kind": "revisit", "mode": "revisit", "parents": ["double_gaussian"]},
        {"idea-1": node},
        {},
        4,
        (("b + a*cos(theta)", "fitted"), ("b + a/(s - c)", "fit_failed")),
        scale=SCALE,
    )
    return new, revisit


def test_the_new_prompt_states_the_limits_the_grammar_enforces():
    for prompt in _v2_prompts():
        assert f"at most {grammar.MAX_PARAMETERS} parameters" in prompt
        assert f"{grammar.MAX_CHARACTERS} characters" in prompt
        assert f"between -{grammar.NUMBER_LIMIT} and {grammar.NUMBER_LIMIT}" in prompt


def test_the_new_prompt_explains_the_data_and_the_cost_of_parameters_without_any_score():
    for prompt in _v2_prompts():
        assert "fitted separately to each" in prompt and "every parameter has a cost" in prompt
        assert "0.4" in prompt and "12" in prompt
        assert all(word not in prompt.lower() for word in SCORE_WORDS)


def test_the_new_prompt_gives_stability_rules_and_an_image_checklist():
    for prompt in _v2_prompts():
        assert "cross zero" in prompt and "tanh" in prompt
        assert "anti-preferred" in prompt and "population" in prompt


def test_the_new_revisit_prompt_says_which_tried_equations_could_not_be_fitted():
    _, revisit = _v2_prompts()
    assert "b + a*cos(theta) (fitted)" in revisit
    assert "b + a/(s - c) (could not be fitted)" in revisit


def test_the_example_equation_object_is_valid():
    assert grammar.compile_equation(EXAMPLE).initial


def test_both_variants_are_registered():
    assert set(PROMPTS) == {"v1", "v2"}


def test_the_image_checklist_does_not_claim_every_neuron_is_averaged():
    for prompt in _v2_prompts():
        assert "averaged over all" not in prompt
