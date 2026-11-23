import asyncio
import time
from types import SimpleNamespace

from edgar_study.pipeline import install

X, Y, CLIENT = object(), object(), object()
FIT_KWARGS = {"param_penalty_weight": 0.01, "fit_params": True, "tol": 1e-6, "use_param_estimator": True}


def loss(prediction, target):
    return 0.0


class Fake:
    """Upstream's stage functions with set delays, logging when each event happens."""

    def __init__(self, delays):
        self.delays = delays
        self.events = []
        self.fits = 0

    def log(self, *event):
        self.events.append((*event, time.monotonic()))

    async def generate_new_neuron_model(
        self,
        current_island,
        llm_name,
        client,
        spike_matrix,
        stimuli,
        mode="explore",
        k_max=2,
        temp=1,
        img_dir=None,
    ):
        code = current_island["next"]
        await asyncio.sleep(self.delays[code])
        self.log("proposed", code)
        return code, f"prompt {code}", (None, None)

    async def translate_to_jax(self, code_string, client, llm_name="tiny"):
        await asyncio.sleep(0.01)
        return f"jax {code_string}", SimpleNamespace(code=code_string, translator=llm_name)

    async def generate_new_parameter_estimator(
        self,
        current_island,
        neuron_model_code_string,
        llm_name,
        client,
        spike_matrix,
        stimuli,
        k_max=1,
        temp=1,
        param_estimator_max_lines=100,
        img_dir=None,
    ):
        await asyncio.sleep(0.01)
        return f"estimator {neuron_model_code_string}", SimpleNamespace(
            code=neuron_model_code_string, temp=temp, model=llm_name
        )

    def objective(
        self,
        neuron_model,
        param_estimator,
        loss_func,
        x,
        y,
        param_penalty_weight=0.1,
        fit_params=True,
        tol=1e-2,
        max_iter=1_000,
        use_param_estimator=True,
    ):
        self.fits += 1
        self.events.append(("fit", neuron_model.code, time.monotonic()))
        return (
            neuron_model.code,
            neuron_model.translator,
            param_estimator.temp,
            param_estimator.model,
            max_iter,
        )


async def upstream_loop(engine, codes, estimator_temp=1.5):
    islands = [{"next": code} for code in codes]
    proposals = await asyncio.gather(
        *[
            engine.generate_new_neuron_model(
                island,
                llm_name="big",
                client=CLIENT,
                mode="explore",
                k_max=2,
                temp=1.5,
                spike_matrix=Y,
                stimuli=X,
                img_dir=None,
            )
            for island in islands
        ]
    )
    code_strings = [proposal[0] for proposal in proposals]
    translations = await asyncio.gather(*[engine.translate_to_jax(c, CLIENT, "tiny") for c in code_strings])
    estimators = await asyncio.gather(
        *[
            engine.generate_new_parameter_estimator(
                current_island=island,
                neuron_model_code_string=code,
                llm_name="little",
                client=CLIENT,
                spike_matrix=Y,
                stimuli=X,
                k_max=2,
                temp=estimator_temp,
                param_estimator_max_lines=100,
                img_dir=None,
            )
            for island, code in zip(islands, code_strings, strict=True)
        ]
    )
    return [
        engine.objective(translation[1], estimator[1], loss_func=loss, x=X, y=Y, **FIT_KWARGS)
        for translation, estimator in zip(translations, estimators, strict=True)
    ]


def recorded(translate):
    """Like the run's translation recorder: a wrapper whose signature is (code_string, *args, **kwargs)."""

    async def translate_and_record(code_string, *args, **kwargs):
        return await translate(code_string, *args, **kwargs)

    return translate_and_record


def pipelined(fake):
    engine = SimpleNamespace(
        generate_new_neuron_model=fake.generate_new_neuron_model,
        translate_to_jax=recorded(fake.translate_to_jax),
        generate_new_parameter_estimator=fake.generate_new_parameter_estimator,
        objective=fake.objective,
    )
    install(engine, translator_model="tiny", estimator_model="little", loss_func=loss, fit_kwargs=FIT_KWARGS)
    return engine


DELAYS = {"fast": 0.01, "medium": 0.05, "slow": 0.3}


def test_pipelined_results_match_upstream_order():
    plain = asyncio.run(upstream_loop(Fake(DELAYS), list(DELAYS)))
    fake = Fake(DELAYS)
    piped = asyncio.run(upstream_loop(pipelined(fake), list(DELAYS)))

    assert piped == plain
    assert fake.fits == len(DELAYS)


def test_fast_programs_are_fitted_while_the_slowest_proposal_is_still_running():
    fake = Fake(DELAYS)
    asyncio.run(upstream_loop(pipelined(fake), list(DELAYS)))

    times = {(event, code): t for event, code, t in fake.events}
    assert times[("fit", "fast")] < times[("proposed", "slow")]
    assert times[("fit", "medium")] < times[("proposed", "slow")]


def test_a_call_upstream_makes_differently_runs_normally():
    fake = Fake(DELAYS)
    results = asyncio.run(upstream_loop(pipelined(fake), list(DELAYS), estimator_temp=0.7))

    assert [result[2] for result in results] == [0.7, 0.7, 0.7]
    assert [result[0] for result in results] == list(DELAYS)


def test_the_final_refit_is_never_served_from_the_pipeline():
    fake = Fake(DELAYS)
    engine = pipelined(fake)
    asyncio.run(upstream_loop(engine, list(DELAYS)))
    model = SimpleNamespace(code="fast", translator="tiny")
    estimator = SimpleNamespace(temp=1.5, model="little")

    result = engine.objective(model, estimator, loss_func=loss, x=X, y=Y, max_iter=2_000, **FIT_KWARGS)

    assert result[-1] == 2_000


def test_an_early_call_that_fails_is_retried_as_upstream_would_make_it():
    fake = Fake(DELAYS)
    failures = {"left": 1}
    real_estimator = fake.generate_new_parameter_estimator

    async def flaky(*args, **kwargs):
        if failures["left"]:
            failures["left"] -= 1
            raise ConnectionError("dropped")
        return await real_estimator(*args, **kwargs)

    fake.generate_new_parameter_estimator = flaky
    plain = asyncio.run(upstream_loop(Fake(DELAYS), list(DELAYS)))

    assert asyncio.run(upstream_loop(pipelined(fake), list(DELAYS))) == plain
