import json

import pytest

from edgar_study.frontier.replies import InvalidReply, decode, decode_json, equation_list


def test_fenced_json_is_read_like_plain_json():
    assert decode_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert decode_json(" [1, 2] ") == [1, 2]


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        (None, "empty_reply"),
        ("   ", "empty_reply"),
        ("not json", "invalid_reply_json"),
        ("[NaN]", "nonfinite_reply_number"),
        ("[Infinity]", "nonfinite_reply_number"),
        ("[1e999]", "nonfinite_reply_number"),
    ],
)
def test_unreadable_replies_have_a_stable_reason_code(reply, code):
    with pytest.raises(InvalidReply) as caught:
        decode_json(reply)
    assert caught.value.code == code


def test_a_reply_must_hold_exactly_k_proposals_unless_fewer_are_allowed():
    assert decode(json.dumps([{}, {}]), 2) == [{}, {}]
    for reply in ("[{}]", "[{}, {}, {}]", '{"a": 1}'):
        with pytest.raises(InvalidReply) as caught:
            decode(reply, 2)
        assert caught.value.code == "invalid_reply_count"
    assert equation_list([], 2, exact=False) == []
    with pytest.raises(InvalidReply):
        equation_list([{}, {}, {}], 2, exact=False)
