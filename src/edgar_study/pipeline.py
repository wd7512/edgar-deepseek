"""Start each program's translation, estimator and fit as soon as its own proposal returns.

Upstream's generation loop waits for every proposal, then every translation, then every
estimator, and only then fits the programs one by one. Proposals with reasoning vary widely in
latency, so most programs sit idle while the slowest proposal finishes. This module patches
upstream's stage functions so that, when a proposal returns, its translation and estimator are
requested immediately with exactly the arguments upstream's loop would pass, and the program is
fitted as soon as both arrive.

When upstream later makes those calls itself, a call whose arguments match the early one gets its
result; any other call runs normally. Upstream's own code and arguments decide every result; only
the timing changes. The one visible difference is the order in which upstream's global NumPy random
state is used (island sampling for prompts, cells for compatibility checks).
"""

import asyncio
import contextlib
import inspect
import logging

log = logging.getLogger(__name__)

ESTIMATOR_K_MAX = 2
ESTIMATOR_MAX_LINES = 100
SIMPLE = (str, int, float, bool, type(None), list, tuple, dict)


def _same(a, b):
    return a is b or (isinstance(a, SIMPLE) and isinstance(b, SIMPLE) and type(a) is type(b) and a == b)


def _bind(function, args, kwargs):
    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    return bound


def _matches(call, expected):
    a, b = call.arguments, expected.arguments
    return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)


def _run(function, call):
    return function(*call.args, **call.kwargs)


def _produced(task, function):
    return task.done() and not task.cancelled() and task.exception() is None and task.result()[1] is function


class _Program:
    def __init__(self, translation_call, estimator_call):
        self.translation_call = translation_call
        self.estimator_call = estimator_call
        self.translation = None
        self.estimator = None
        self.fit = None
        self.fit_call = None
        self.translation_taken = False
        self.estimator_taken = False
        self.claimed = False


class Pipeline:
    def __init__(self, engine, translator_model, estimator_model, loss_func, fit_kwargs):
        self.propose = engine.generate_new_neuron_model
        self.translate = engine.translate_to_jax
        self.estimate = engine.generate_new_parameter_estimator
        self.objective = engine.objective
        self.translator_model = translator_model
        self.estimator_model = estimator_model
        self.loss_func = loss_func
        self.fit_kwargs = fit_kwargs
        self.programs = []

    async def proposal(self, *args, **kwargs):
        result = await self.propose(*args, **kwargs)
        code = result[0] if result else None
        if code is not None:
            self._start(code, _bind(self.propose, args, kwargs).arguments)
        return result

    def _start(self, code, proposal):
        translation_call = _bind(self.translate, (code, proposal["client"], self.translator_model), {})
        estimator_call = _bind(
            self.estimate,
            (),
            {
                "current_island": proposal["current_island"],
                "neuron_model_code_string": code,
                "llm_name": self.estimator_model,
                "client": proposal["client"],
                "spike_matrix": proposal["spike_matrix"],
                "stimuli": proposal["stimuli"],
                "k_max": ESTIMATOR_K_MAX,
                "temp": proposal["temp"],
                "param_estimator_max_lines": ESTIMATOR_MAX_LINES,
                "img_dir": None,
            },
        )
        program = _Program(translation_call, estimator_call)
        program.translation = asyncio.ensure_future(_run(self.translate, translation_call))
        program.estimator = asyncio.ensure_future(_run(self.estimate, estimator_call))
        program.fit = asyncio.ensure_future(self._fit(program, proposal))
        self.programs.append(program)

    async def _fit(self, program, proposal):
        try:
            _, model = await program.translation
            _, estimator = await program.estimator
        except Exception:
            return None
        if model is None or estimator is None or program.claimed:
            return None
        call = _bind(
            self.objective,
            (model, estimator),
            {
                "loss_func": self.loss_func,
                "x": proposal["stimuli"],
                "y": proposal["spike_matrix"],
                **self.fit_kwargs,
            },
        )
        try:
            result = _run(self.objective, call)
        except Exception:
            log.exception("Early fit failed; upstream will fit this program itself.")
            return None
        program.fit_call = call
        return result

    async def translation(self, *args, **kwargs):
        call = _bind(self.translate, args, kwargs)
        program = self._take(call, "translation")
        if program is not None:
            with contextlib.suppress(Exception):
                return await program.translation
        return await self.translate(*args, **kwargs)

    async def estimator(self, *args, **kwargs):
        call = _bind(self.estimate, args, kwargs)
        program = self._take(call, "estimator")
        if program is not None:
            with contextlib.suppress(Exception):
                return await program.estimator
        return await self.estimate(*args, **kwargs)

    def _take(self, call, stage):
        for program in self.programs:
            if not getattr(program, f"{stage}_taken") and _matches(call, getattr(program, f"{stage}_call")):
                setattr(program, f"{stage}_taken", True)
                return program
        return None

    def fitted(self, *args, **kwargs):
        call = _bind(self.objective, args, kwargs)
        for program in list(self.programs):
            if not _produced(program.translation, call.arguments["neuron_model"]):
                continue
            if not _produced(program.estimator, call.arguments["param_estimator"]):
                continue
            self.programs.remove(program)
            program.claimed = True
            if program.fit.done() and program.fit_call is not None and _matches(call, program.fit_call):
                return program.fit.result()
            break
        return self.objective(*args, **kwargs)


def install(hypothesis_engine, translator_model, estimator_model, loss_func, fit_kwargs):
    """Patch upstream's stage functions so each program moves on as soon as its proposal returns."""
    pipeline = Pipeline(hypothesis_engine, translator_model, estimator_model, loss_func, fit_kwargs)
    hypothesis_engine.generate_new_neuron_model = pipeline.proposal
    hypothesis_engine.translate_to_jax = pipeline.translation
    hypothesis_engine.generate_new_parameter_estimator = pipeline.estimator
    hypothesis_engine.objective = pipeline.fitted
    return pipeline
