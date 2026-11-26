from edgar_study.frontier.ideas import Idea
from edgar_study.frontier.prompts import generated_prompt, idea_prompt, tool_note


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
