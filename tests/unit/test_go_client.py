"""Offline checks for the Go client, ledger, paid-model guard and tier mapping."""

import asyncio
import base64
import json
import os
import types

import httpx

from edgar_study.engine import install
from edgar_study.frontier.images import FrontierImages
from edgar_study.go_client import BudgetExhausted, GoClient, Ledger, role


def fake_transport(replies, bodies):
    def handler(request):
        bodies.append(json.loads(request.content))
        status, payload = replies.pop(0)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def ok(text, list_cost=0.0):
    return 200, {
        "model": "m",
        "choices": [{"message": {"content": text}}],
        "usage": {"completion_tokens": 3, "opencode": {"list_cost": list_cost}},
    }


def fake_upstream():
    """Just enough of the upstream modules for engine.install to patch."""

    async def coroutine(*args, **kwargs):
        return None, None

    utils = types.SimpleNamespace(
        create_program_prompt=None, create_parameter_estimator_prompt=None, create_linked_prompt=None
    )
    engine = types.SimpleNamespace(
        genai=types.SimpleNamespace(Client=None),
        translate_to_jax=coroutine,
        generate_new_neuron_model=coroutine,
        generate_new_parameter_estimator=coroutine,
    )
    return engine, utils


async def check_client(directory):
    os.environ.pop("OPENCODE_API_KEY_FILE", None)
    os.environ["OPENCODE_API_KEY"] = "test-key"
    ledger = Ledger(directory / "spend.jsonl", cap=1.0)
    go = GoClient(directory, ledger, session_id="test", free_models=["hy4-flash-free"])
    go.retry_delay_s = 0
    replies = [
        ok("```python\ndef neuron_model(theta): return theta\n```"),
        (500, {"error": "boom"}),
        ok("second try"),
        (401, {"error": "bad key"}),
        ok(""),
    ]
    bodies = []
    go._http = httpx.AsyncClient(
        base_url="https://example.invalid", transport=fake_transport(replies, bodies)
    )

    # Upstream's tiers and temperatures are mapped once, in the patched call_llm_async.
    engine, utils = fake_upstream()
    install(engine, utils, go, default_effort="none", temperature_scale=0.5)
    role.set("translation")
    assert "neuron_model" in await utils.call_llm_async("prompt", model_name="hy4-flash-free", temperature=0)
    assert (
        await utils.call_llm_async(
            "prompt", model_name="hy4-flash-free#high", temperature=1.8, img_bytes=b"png"
        )
        == "second try"
    )  # retried 5xx
    assert await go.chat("hy4-flash-free", "prompt") is None  # 401 is not retried
    assert await go.chat("hy4-flash-free", "prompt") is None  # empty content counts as failure
    assert bodies[0]["reasoning_effort"] == "none" and bodies[0]["temperature"] == 0
    assert bodies[1]["model"] == "hy4-flash-free" and bodies[1]["reasoning_effort"] == "high"
    assert bodies[1]["temperature"] == 0.9
    try:
        await go.chat("deepseek-v4.1-flash", "prompt")
        raise AssertionError("paid model should be refused without --allow-paid")
    except BudgetExhausted:
        pass

    rows = [json.loads(line) for line in (directory / "api_calls.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == [200, 500, 200, 401, 200]
    assert rows[0]["role"] == "translation" and rows[1]["image"] is True
    assert rows[1]["temperature_upstream"] == 1.8
    assert "test-key" not in (directory / "api_calls.jsonl").read_text()

    # The ledger refuses new requests once list-equivalent spend reaches the cap, across instances.
    ledger.add(1.0, model="x")
    try:
        Ledger(directory / "spend.jsonl", cap=1.0).check()
        raise AssertionError("ledger should be exhausted")
    except BudgetExhausted:
        pass
    await go.close()


def test_client_ledger_and_tier_mapping(tmp_path):
    asyncio.run(check_client(tmp_path))


def test_sends_eighty_images_with_one_prompt_in_order(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY_FILE", raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    bodies = []
    images = tuple(f"attachment {i}".encode() for i in range(80))

    async def run():
        go = GoClient(tmp_path, Ledger(tmp_path / "spend.jsonl"), "test", ["hy4-flash-free"])
        await go.close()
        go._http = httpx.AsyncClient(
            base_url="https://example.invalid", transport=fake_transport([ok("equations"), ok("new")], bodies)
        )
        try:
            batch = FrontierImages(tuple(range(10)), ("a", "b", "c", "d"), images)
            reply = await batch.submit(go, "Combine the ideas.")
            assert reply == "equations"
            history = [(reply, "Suggest another equation."), ("more", "Check this one.")]
            return await batch.submit(go, "Combine the ideas.", history=history)
        finally:
            await go.close()

    assert asyncio.run(run()) == "new"
    content = bodies[0]["messages"][0]["content"]
    assert bodies[0]["model"] == "hy4-flash-free"
    assert content[0] == {
        "type": "text",
        "text": "Combine the ideas.\nParent image blocks, in order: a, b, c, d."
        "\nEach neuron has two images: response and fit, then residuals by angle.",
    }
    assert [base64.b64decode(p["image_url"]["url"].split(",", 1)[1]) for p in content[1:]] == list(images)
    assert bodies[1]["messages"] == [
        bodies[0]["messages"][0],
        {"role": "assistant", "content": "equations"},
        {"role": "user", "content": "Suggest another equation."},
        {"role": "assistant", "content": "more"},
        {"role": "user", "content": "Check this one."},
    ]
    rows = [json.loads(line) for line in (tmp_path / "api_calls.jsonl").read_text().splitlines()]
    assert all(row["image_count"] == 80 and row["image"] is True for row in rows)


def client(directory, handler, max_concurrency=16):
    os.environ.pop("OPENCODE_API_KEY_FILE", None)
    os.environ["OPENCODE_API_KEY"] = "test-key"
    go = GoClient(
        directory,
        Ledger(directory / "spend.jsonl"),
        session_id="test",
        free_models=["hy4-flash-free"],
        max_concurrency=max_concurrency,
    )
    go._http = httpx.AsyncClient(base_url="https://example.invalid", transport=httpx.MockTransport(handler))
    return go


async def check_queue(directory):
    release = asyncio.Event()

    async def handler(request):
        await release.wait()
        status, payload = ok("reply")
        return httpx.Response(status, json=payload)

    go = client(directory, handler, max_concurrency=1)
    calls = [asyncio.ensure_future(go.chat("hy4-flash-free", "prompt")) for _ in range(3)]
    await asyncio.sleep(0.05)
    assert go.queued == 2
    release.set()
    assert await asyncio.gather(*calls) == ["reply"] * 3
    assert go.queued == 0
    await go.close()


def test_the_client_counts_requests_waiting_for_a_slot(tmp_path):
    asyncio.run(check_queue(tmp_path))


async def check_blocked(directory):
    go = client(directory, None)

    async def handler(request):
        go.blocked_s += 2.5
        status, payload = ok("reply")
        return httpx.Response(status, json=payload)

    go._http = httpx.AsyncClient(base_url="https://example.invalid", transport=httpx.MockTransport(handler))
    await go.chat("hy4-flash-free", "prompt")
    await go.close()
    row = json.loads((directory / "api_calls.jsonl").read_text().splitlines()[-1])
    assert row["blocked_s"] == 2.5


def test_each_call_records_how_long_the_event_loop_was_blocked_during_it(tmp_path):
    asyncio.run(check_blocked(tmp_path))
