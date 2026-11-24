"""Decode a NEAT genome into an EDGAR neuron model and its parameter estimator.

A genome is a feed-forward network over periodic inputs, `sin(theta)`, `cos(theta)` and a
constant. Every enabled weight and bias becomes a named parameter of the generated
`neuron_model`, so the existing `hypothesis_engine.objective` fits them per cell exactly as it
fits an equation's parameters. The generated estimator returns the genome's weights, shifted so
the output starts at the cell's mean response; gradient fitting then specialises them.

The output node always uses a softplus, which keeps the predicted rate non-negative without a
dead gradient when the pre-activation is negative. Hidden nodes use the activation NEAT assigns.
"""

import heapq
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import neat

INPUT_SIGNALS = ("jnp.sin(theta)", "jnp.cos(theta)", "jnp.ones_like(theta)")
ACTIVATIONS = ("identity", "tanh", "sin", "cos", "gauss", "square", "abs", "softplus")
PREACTIVATION_BOUND = 30.0

CONFIG_TEMPLATE = """\
[NEAT]
fitness_criterion = max
fitness_threshold = 1e9
pop_size = {population}
reset_on_extinction = False
no_fitness_termination = True

[DefaultGenome]
activation_default = tanh
activation_mutate_rate = 0.3
activation_options = {activations}
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
bias_init_mean = 0.0
bias_init_stdev = 0.5
bias_max_value = 10.0
bias_min_value = -10.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob = 0.2
conn_delete_prob = 0.05
enabled_default = True
enabled_mutate_rate = 0.01
feed_forward = True
initial_connection = full_direct
node_add_prob = 0.3
node_delete_prob = 0.05
num_hidden = 0
num_inputs = {inputs}
num_outputs = 1
response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0
weight_init_mean = 0.0
weight_init_stdev = 0.5
weight_max_value = 10.0
weight_min_value = -10.0
weight_mutate_power = 0.5
weight_mutate_rate = 0.8
weight_replace_rate = 0.1
single_structural_mutation = false
structural_mutation_surer = default
bias_init_type = gaussian
response_init_type = gaussian
weight_init_type = gaussian

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 15
species_elitism = 1

[DefaultReproduction]
elitism = 1
survival_threshold = 0.2
min_species_size = 1
"""


@dataclass(frozen=True)
class Program:
    """Generated EDGAR program source and the shape of its parameter vector."""

    model_code: str
    estimator_code: str
    parameter_names: tuple[str, ...]
    parameter_defaults: tuple[float, ...]
    output_bias_index: int
    hidden_nodes: int
    activations: tuple[str, ...]
    connections: int

    @property
    def n_params(self):
        return len(self.parameter_names)


def make_config(population, inputs=None):
    """Build a NEAT config in a temporary file and parse it with neat-python."""
    inputs = len(INPUT_SIGNALS) if inputs is None else inputs
    text = CONFIG_TEMPLATE.format(
        population=population,
        inputs=inputs,
        activations=" ".join(ACTIVATIONS),
    )
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as stream:
        stream.write(text)
        path = Path(stream.name)
    try:
        return neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(path),
        )
    finally:
        path.unlink(missing_ok=True)


def decode(genome, config):
    """Turn a genome into `neuron_model` and `parameter_estimator` source, deterministically."""
    genome_config = config.genome_config
    input_keys = tuple(genome_config.input_keys)
    output_keys = tuple(genome_config.output_keys)
    if len(input_keys) != len(INPUT_SIGNALS):
        raise ValueError(f"Expected {len(INPUT_SIGNALS)} inputs, got {len(input_keys)}.")
    if len(output_keys) != 1:
        raise ValueError(f"Expected one output, got {len(output_keys)}.")

    enabled = {connection.key: connection for connection in genome.connections.values() if connection.enabled}
    incoming = defaultdict(list)
    for (input_key, output_key), connection in enabled.items():
        incoming[output_key].append((input_key, connection))

    needed = set(output_keys)
    stack = list(output_keys)
    while stack:
        node = stack.pop()
        for input_key, _ in incoming.get(node, ()):
            if input_key not in needed:
                needed.add(input_key)
                if input_key not in input_keys:
                    stack.append(input_key)

    used = [
        connection
        for connection in enabled.values()
        if connection.key[0] in needed and connection.key[1] in needed
    ]
    used.sort(key=lambda connection: connection.key)
    nodes = sorted(
        (key for key in needed if key not in input_keys),
        key=lambda key: key,
    )
    order = _topological_order(nodes, used)

    weight_names = {connection.key: f"w{index}" for index, connection in enumerate(used)}
    bias_names = {key: f"b{key}" for key in nodes}
    parameters = [(weight_names[connection.key], connection.weight) for connection in used]
    parameters += [(bias_names[key], genome.nodes[key].bias) for key in nodes]

    hidden = [key for key in order if key not in output_keys]
    lines = [
        "import jax.numpy as jnp",
        "",
        f"def neuron_model(theta, {_signature(parameters)}):",
    ]
    for index in range(len(input_keys)):
        lines.append(f"    x{index} = {INPUT_SIGNALS[index]}")
    for key in order:
        terms = _terms(key, incoming, weight_names, input_keys)
        terms = terms or ["jnp.zeros_like(theta)"]
        inner = f"{bias_names[key]} + ({' + '.join(terms)})"
        inner = f"jnp.clip({inner}, -{PREACTIVATION_BOUND}, {PREACTIVATION_BOUND})"
        if key in output_keys:
            lines.append(f"    out = {inner}")
        else:
            lines.append(f"    h{key} = {_activate(genome.nodes[key].activation, inner)}")
    lines.append("    return jnp.logaddexp(0.0, out)")

    output_bias_index = [name for name, _ in parameters].index(bias_names[output_keys[0]])
    estimator = [
        "import numpy as np",
        "",
        "def parameter_estimator(theta, spike_counts):",
        f"    params = np.array({_array(parameters)}, dtype=np.float32)",
        f"    params[{output_bias_index}] += float(np.mean(spike_counts))",
        "    return params",
    ]
    return Program(
        model_code="\n".join(lines) + "\n",
        estimator_code="\n".join(estimator) + "\n",
        parameter_names=tuple(name for name, _ in parameters),
        parameter_defaults=tuple(float(value) for _, value in parameters),
        output_bias_index=output_bias_index,
        hidden_nodes=len(hidden),
        activations=tuple(genome.nodes[key].activation for key in hidden),
        connections=len(used),
    )


def load(code, name):
    """Execute generated source in a fresh namespace and return `name`."""
    namespace = {}
    exec(code, namespace)
    return namespace[name]


def _topological_order(nodes, connections):
    edges = defaultdict(list)
    indegree = dict.fromkeys(nodes, 0)
    for connection in connections:
        input_key, output_key = connection.key
        if input_key in indegree:
            edges[input_key].append(output_key)
            indegree[output_key] += 1
    ready = [key for key in nodes if indegree[key] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for output in edges[node]:
            indegree[output] -= 1
            if indegree[output] == 0:
                heapq.heappush(ready, output)
    if len(order) != len(nodes):
        raise ValueError("Genome is not feed-forward.")
    return order


def _terms(node, incoming, weight_names, input_keys):
    terms = []
    for input_key, connection in sorted(incoming.get(node, ()), key=lambda pair: pair[1].key):
        source = f"x{input_keys.index(input_key)}" if input_key in input_keys else f"h{input_key}"
        terms.append(f"{weight_names[connection.key]} * {source}")
    return terms


def _activate(name, inner):
    if name == "identity":
        return f"({inner})"
    if name == "tanh":
        return f"jnp.tanh({inner})"
    if name == "sin":
        return f"jnp.sin({inner})"
    if name == "cos":
        return f"jnp.cos({inner})"
    if name == "gauss":
        return f"jnp.exp(-0.5 * ({inner}) ** 2)"
    if name == "square":
        return f"({inner}) ** 2"
    if name == "abs":
        return f"jnp.abs({inner})"
    if name == "softplus":
        return f"jnp.logaddexp(0.0, {inner})"
    raise ValueError(f"Unsupported activation {name!r}.")


def _signature(parameters):
    return ", ".join(f"{name}={value:.8g}" for name, value in parameters)


def _array(parameters):
    return "[" + ", ".join(f"{value:.8g}" for _, value in parameters) + "]"
