"""Offline checks for turning NEAT genomes into EDGAR neuron models and estimators."""

import inspect
import math

import jax
import jax.numpy as jnp
import neat
import numpy as np

from edgar_study import neat_programs


def population(seed=0):
    config = neat_programs.make_config(population=6)
    return neat.Population(config, seed=seed), config


def test_decode_gives_a_periodic_nonnegative_curve():
    population_, config = population()
    genome = next(iter(population_.population.values()))
    program = neat_programs.decode(genome, config)

    model = neat_programs.load(program.model_code, "neuron_model")
    names = [name for name in inspect.signature(model).parameters if name != "theta"]
    assert len(names) == program.n_params
    assert program.n_params >= 4

    theta = jnp.linspace(0.0, 2 * math.pi, 40, endpoint=False)
    params = jnp.asarray([inspect.signature(model).parameters[name].default for name in names])
    fitted = jax.jit(model)(theta, *params)
    assert fitted.shape == theta.shape
    assert np.all(np.isfinite(fitted))
    assert np.all(np.asarray(fitted) >= 0)
    assert np.allclose(jax.jit(model)(theta + 2 * math.pi, *params), fitted, atol=1e-4)


def test_estimator_returns_the_genome_weights_with_the_cell_mean_shifted_into_the_output_bias():
    population_, config = population()
    genome = next(iter(population_.population.values()))
    program = neat_programs.decode(genome, config)

    model = neat_programs.load(program.model_code, "neuron_model")
    estimator = neat_programs.load(program.estimator_code, "parameter_estimator")
    defaults = np.asarray(
        [
            parameter.default
            for parameter in inspect.signature(model).parameters.values()
            if parameter.name != "theta"
        ]
    )
    theta = np.linspace(0.0, 2 * math.pi, 20, endpoint=False)
    counts = np.linspace(0.0, 4.0, 20)
    estimated = np.asarray(estimator(theta, counts))

    assert estimated.shape == (program.n_params,)
    assert np.all(np.isfinite(estimated))
    expected = defaults.copy()
    expected[program.output_bias_index] += float(np.mean(counts))
    assert np.allclose(estimated, expected, atol=1e-6)


def test_decoding_is_deterministic_and_source_identifies_the_weights():
    population_, config = population()
    genome = next(iter(population_.population.values()))
    first = neat_programs.decode(genome, config)
    second = neat_programs.decode(genome, config)
    assert first == second

    genome.connections[next(iter(genome.connections))].weight += 0.5
    changed = neat_programs.decode(genome, config)
    assert changed.model_code != first.model_code
    assert changed.n_params == first.n_params


def test_a_new_hidden_node_changes_the_architecture_and_parameter_count():
    population_, config = population()
    genome = next(iter(population_.population.values()))
    before = neat_programs.decode(genome, config)

    original_weight = next(iter(genome.connections.values())).weight
    genome.mutate_add_node(config.genome_config)
    after = neat_programs.decode(genome, config)

    assert before.hidden_nodes == 0
    assert after.hidden_nodes == 1
    assert after.n_params > before.n_params
    assert original_weight in [connection.weight for connection in genome.connections.values()]
    model = neat_programs.load(after.model_code, "neuron_model")
    theta = jnp.linspace(0.0, 2 * math.pi, 16, endpoint=False)
    params = [
        parameter.default
        for parameter in inspect.signature(model).parameters.values()
        if parameter.name != "theta"
    ]
    assert jax.jit(model)(theta, *params).shape == theta.shape


def test_a_genome_without_enabled_connections_predicts_an_array():
    population_, config = population()
    genome = next(iter(population_.population.values()))
    for connection in genome.connections.values():
        connection.enabled = False
    program = neat_programs.decode(genome, config)

    assert program.connections == 0
    assert program.n_params == 1
    model = neat_programs.load(program.model_code, "neuron_model")
    theta = jnp.linspace(0.0, 2 * math.pi, 16, endpoint=False)
    fitted = jax.jit(model)(theta, program.parameter_defaults[0])

    assert fitted.shape == theta.shape
    assert np.allclose(fitted, fitted[0])


def test_generated_source_exposes_no_globals_beyond_jax_and_numpy():
    population_, config = population()
    genome = next(iter(population_.population.values()))
    program = neat_programs.decode(genome, config)
    assert "jnp." in program.model_code
    assert program.model_code.startswith("import jax.numpy as jnp")
    assert program.estimator_code.startswith("import numpy as np")
