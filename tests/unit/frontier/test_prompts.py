from edgar_study.frontier.ideas import Idea
from edgar_study.frontier.prompts import generated_prompt


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
