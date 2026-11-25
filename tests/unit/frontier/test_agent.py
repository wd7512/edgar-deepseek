"""The batch conversation: checks, repairs, unreadable replies and patience; and the score-free check tool."""

import asyncio
import json

import pytest

from edgar_study.frontier.agent import check_proposal, check_report, check_request, converse
from edgar_study.frontier.grammar import compile_equation
from edgar_study.frontier.replies import InvalidReply, decode
from edgar_study.frontier.warm import Parent, declared_probes

SCORE_WORDS = ("mse", "score", "loss", "ranking", "best", "error")


def eq(text, **initial):
    return {"equation": text, "initial": initial or {"a": 1.0, "b": 1.0}}


GOOD = eq("b + a*cos(theta)")
OTHER = eq("b + a*sin(theta)")
BAD = eq("y = a*cos(theta)")


class Proposer:
    def __init__(self, *replies):
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.histories = []

    async def submit(self, batch, prompt):
        return self.replies.pop(0)

    async def retry(self, batch, prompt, history):
        self.histories.append(history)
        return self.replies.pop(0)


class Session:
    """Equations whose text contains 'cos' or 'sin' fit; anything else fails; reads JSON arrays."""

    def __init__(self):
        self.submitted = []
        self.checked = []

    def decode(self, reply, expected):
        return decode(reply, expected) if isinstance(expected, int) else json.loads(reply)

    def submit(self, proposal, replaces, turn):
        self.submitted.append((proposal["equation"], replaces, turn))
        return (
            None
            if "theta)" in proposal["equation"] and "=" not in proposal["equation"]
            else ("invalid", "no")
        )

    def check(self, proposals):
        self.checked.append(proposals)
        return "Check results: fine."


def talk(proposer, session=None, patience=3, k=2):
    session = session or Session()
    return asyncio.run(converse(proposer, None, "prompt", session, k=k, patience=patience)), session


def test_a_reply_whose_equations_all_fit_ends_without_follow_ups():
    proposer = Proposer([GOOD, OTHER])
    transcript, session = talk(proposer)

    assert transcript["ended"] == "done" and transcript["turns"] == []
    assert session.submitted == [(GOOD["equation"], None, 0), (OTHER["equation"], None, 0)]


def test_a_failed_equation_is_replaced_in_its_position_until_it_fits():
    proposer = Proposer([GOOD, BAD], [BAD], [OTHER])
    transcript, session = talk(proposer)

    assert transcript["ended"] == "done"
    assert [turn["kind"] for turn in transcript["turns"]] == ["replace", "replace"]
    assert "Proposal 2: invalid — no" in transcript["turns"][0]["instruction"]
    assert "exactly 1 replacement" in transcript["turns"][0]["instruction"]
    assert session.submitted[-2:] == [(BAD["equation"], 2, 1), (OTHER["equation"], 2, 2)]


def test_the_model_can_check_equations_before_answering_and_a_check_is_not_a_submission():
    proposer = Proposer({"check": [BAD]}, [GOOD])
    transcript, session = talk(proposer)

    assert session.checked == [[BAD]]
    assert session.submitted == [(GOOD["equation"], None, 1)]
    assert transcript["turns"][0]["kind"] == "check"
    assert transcript["turns"][0]["instruction"].startswith("Check results: fine.")


def test_the_whole_conversation_is_sent_back_each_turn():
    proposer = Proposer([BAD], [BAD], [GOOD])
    talk(proposer)

    assert [len(h) for h in proposer.histories] == [1, 2]
    assert proposer.histories[1][0] == proposer.histories[0][0]
    assert proposer.histories[1][1][0] == json.dumps([BAD])


def test_patience_bounds_the_follow_ups_and_the_last_turn_says_so():
    proposer = Proposer([BAD], [BAD], [BAD], [GOOD])
    transcript, _ = talk(proposer, patience=2)

    assert len(transcript["turns"]) == 2
    assert transcript["ended"] == "out_of_patience_after_replace"
    assert "last turn" in transcript["turns"][-1]["instruction"]
    assert "last turn" not in transcript["turns"][0]["instruction"]


def test_with_no_patience_the_first_reply_is_all_there_is():
    transcript, session = talk(Proposer([BAD]), patience=0)

    assert transcript["turns"] == [] and transcript["ended"] == "out_of_patience_after_replace"
    assert len(session.submitted) == 1


def test_a_check_on_the_last_turn_ends_the_batch_without_an_answer():
    transcript, session = talk(Proposer({"check": [GOOD]}, {"check": [GOOD]}), patience=1)

    assert transcript["ended"] == "check_without_answer"
    assert session.submitted == []


def test_an_unreadable_reply_is_asked_for_again_with_the_reason():
    proposer = Proposer("not json", [GOOD])

    class Reading(Session):
        def decode(self, reply, expected):
            try:
                return json.loads(reply)
            except ValueError as exc:
                raise InvalidReply("invalid_reply_json", "Reply is not readable JSON.") from exc

    transcript, session = talk(proposer, Reading())

    assert transcript["turns"][0]["kind"] == "unreadable"
    assert "invalid_reply_json" in transcript["turns"][0]["instruction"]
    assert session.submitted == [(GOOD["equation"], None, 1)]


def test_a_reply_with_no_equations_asks_for_fresh_ones():
    transcript, _ = talk(Proposer([], [GOOD]))

    assert transcript["turns"][0]["kind"] == "fresh"
    assert "up to 2 equation objects" in transcript["turns"][0]["instruction"]


def test_follow_ups_never_mention_fit_quality():
    transcript, _ = talk(Proposer({"check": [GOOD]}, [BAD], [BAD], [GOOD]))
    text = " ".join(turn["instruction"] for turn in transcript["turns"]).lower()

    assert all(word not in text for word in SCORE_WORDS)


def test_only_an_object_with_just_a_check_list_is_a_check_request():
    assert check_request('{"check": [1]}') == [1]
    assert check_request('{"check": "x"}') == []
    assert check_request('{"check": [], "idea": {}}') is None
    assert check_request("[]") is None and check_request("nonsense") is None


GAUSS = "b + a1*exp(-0.5*(wrap(theta-m)/s)**2) + a2*exp(-0.5*(wrap(theta-m-pi)/s)**2)"
START = {"b": 0.3, "a1": 2.0, "a2": 1.0, "m": 1.0, "s": 0.4}


def gauss_parent():
    equation = compile_equation({"equation": GAUSS, "initial": START})
    return Parent("double_gaussian", tuple(equation.initial), equation.model, declared_probes(equation), None)


def test_a_statement_instead_of_an_expression_is_reported_as_unusable():
    result = check_proposal(BAD)
    assert result["problems"] and result["problems"][0].startswith("invalid_syntax")


def test_repeats_and_renamed_duplicates_are_reported():
    known = compile_equation(GOOD)
    assert "exact_repeat" in check_proposal(GOOD, texts=[known.text])["problems"][0]
    renamed = eq("q + p*cos(theta)", p=1.0, q=1.0)
    assert "duplicate" in check_proposal(renamed, keys=[known.key])["problems"][0]


def test_values_that_overflow_at_the_start_are_reported():
    result = check_proposal(eq("a*exp(60*sin(theta)) + b"))
    assert result["problems"][0].startswith("overflow_start")


def test_an_undefined_slope_at_the_start_is_a_note_not_a_problem():
    result = check_proposal(eq("b + sqrt(a)*cos(theta)", a=0.0, b=1.0))
    assert result["problems"] == [] and "slope" in result["notes"][0]


def test_the_nesting_map_is_checked_and_a_working_one_is_offered():
    shifted = "b + a1*exp(-0.5*(wrap(theta-(m+d1))/s)**2) + a2*exp(-0.5*(wrap(theta-(m+pi+d2))/s)**2)"
    proposal = {
        "equation": shifted,
        "initial": {**START, "d1": 0.1, "d2": 0.1},
        "nesting": {"double_gaussian": {"b": "b", "a1": "a1", "a2": "a2", "m": "m", "s": "s"}},
    }
    (note,) = check_proposal(proposal, parents=[gauss_parent()])["notes"]
    assert "does not reproduce it; this one does" in note
    assert json.loads(note.split("this one does: ")[1].rstrip("."))["d1"] == 0.0


def test_the_report_lists_each_equation_in_order():
    report = check_report([{"problems": [], "notes": ["n"]}, {"problems": ["p"], "notes": []}])
    assert report.splitlines() == [
        "Check results:",
        "- Equation 1: usable. n",
        "- Equation 2: cannot be used: p",
    ]


@pytest.mark.parametrize(
    "proposal", [eq("b + a*cos(theta)"), eq("b + a*exp(-((wrap(theta-m))/s)**2)", a=1.0, b=0.0, m=1.0, s=0.5)]
)
def test_well_formed_equations_pass_the_check(proposal):
    assert check_proposal(proposal)["problems"] == []
