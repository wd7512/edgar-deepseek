"""Compile model-proposed equations from a small safe grammar."""

import ast
import operator
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


def _soft_cutoff(left, right):
    if jnp.iscomplexobj(left) or jnp.iscomplexobj(right):
        raise InvalidEquation("nonreal_expression", "Cutoffs require real-valued inputs.")
    return jax.nn.sigmoid(20 * (right - left))


def _soft_cutoff_reverse(left, right):
    return _soft_cutoff(right, left)


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Lt: _soft_cutoff,
    ast.LtE: _soft_cutoff,
    ast.Gt: _soft_cutoff_reverse,
    ast.GtE: _soft_cutoff_reverse,
}
CUTOFFS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
FUNCTIONS = {name: getattr(jnp, name) for name in ("sin", "cos", "exp", "tanh", "abs", "sqrt")}
FUNCTIONS["wrap"] = lambda x: jnp.arctan2(jnp.sin(x), jnp.cos(x))


@dataclass(frozen=True)
class Equation:
    text: str
    initial: dict[str, float]
    key: str
    model: object
    estimator: object
    canonical_names: tuple[str, ...]
    nesting: object


class InvalidEquation(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


SYMBOLS = {
    ast.Mod: "%",
    ast.FloorDiv: "//",
    ast.MatMult: "@",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
}
MAX_PARAMETERS = 12
MAX_CHARACTERS = 600
MAX_NODES = 260
NUMBER_LIMIT = 100
ALLOWED = "Use + - * / **, sin, cos, exp, tanh, abs, sqrt, wrap and one comparison with <, <=, > or >=."


def _unsupported(node):
    """Name the construct the grammar does not accept, and what to use instead."""
    if isinstance(node, ast.IfExp):
        detail = (
            "'x if condition else y' is not supported; multiply a term by a comparison such as (theta < c)."
        )
    elif isinstance(node, ast.BinOp):
        symbol = SYMBOLS.get(type(node.op), type(node.op).__name__)
        detail = f"Operator '{symbol}' is not supported; use wrap() for periodic distances."
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        detail = f"Use plain function names, not {ast.unparse(node.func)}."
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in FUNCTIONS:
        detail = f"Function '{node.func.id}' is not supported."
    elif isinstance(node, ast.Call):
        detail = "Each function takes exactly one argument."
    elif isinstance(node, ast.Compare):
        detail = "Use a single comparison with <, <=, > or >=."
    elif isinstance(node, ast.Constant):
        detail = "Numbers must be between -100 and 100."
    else:
        detail = f"{type(node).__name__} is not supported."
    return f"{detail} {ALLOWED}"


def compile_equation(proposal):
    """Compile a small numeric expression tree without executing generated Python."""
    if not isinstance(proposal, dict):
        raise InvalidEquation("invalid_equation_object", "Expected an equation object.")
    text, initial = proposal.get("equation"), proposal.get("initial")
    if not isinstance(text, str) or not isinstance(initial, dict):
        raise InvalidEquation(
            "invalid_equation_object", "Expected an equation string and a parameter object."
        )
    if len(text) > MAX_CHARACTERS:
        raise InvalidEquation("equation_too_complex", f"Equation exceeds {MAX_CHARACTERS} characters.")
    if not 1 <= len(initial) <= MAX_PARAMETERS:
        raise InvalidEquation("invalid_parameter_count", "Expected one to twelve free parameters.")
    for name, value in initial.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or not name.isascii()
            or name in {"pi", "theta", *FUNCTIONS}
            or type(value) not in (float, int)
            or not -NUMBER_LIMIT <= value <= NUMBER_LIMIT
        ):
            raise InvalidEquation("invalid_parameter", "Invalid parameter name or initial value.")
    try:
        expr = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise InvalidEquation("invalid_syntax", "Equation is not a valid expression.") from exc
    except RecursionError as exc:
        raise InvalidEquation("equation_too_complex", "Equation is nested too deeply.") from exc
    if sum(1 for _ in ast.walk(expr)) > MAX_NODES:
        raise InvalidEquation("equation_too_complex", "Equation exceeds the expression complexity limit.")
    used = set()

    def build(node):
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in (float, int)
            and -NUMBER_LIMIT <= node.value <= NUMBER_LIMIT
        ):
            return ("number", float(node.value))
        if isinstance(node, ast.Name):
            if node.id not in {"pi", "theta", *initial}:
                raise InvalidEquation(
                    "unknown_name",
                    f"Undefined name {node.id!r}; declare it in initial or use theta, pi and the declared parameters.",
                )
            used.add(node.id)
            return ("name", node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return (type(node.op), build(node.left), build(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CUTOFFS:
            return (type(node.ops[0]), build(node.left), build(node.comparators[0]))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return ("neg", build(node.operand)) if isinstance(node.op, ast.USub) else build(node.operand)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS
            and len(node.args) == 1
            and not node.keywords
        ):
            return ("call", node.func.id, build(node.args[0]))
        raise InvalidEquation("unsupported_operator", _unsupported(node))

    tree = build(expr)
    if set(initial) != used - {"pi", "theta"} or "theta" not in used:
        raise InvalidEquation(
            "parameter_mismatch", "Equation must use theta and exactly the listed parameters."
        )

    def shape(tree):
        op, *parts = tree
        if op == "name" and parts[0] in initial:
            return "parameter"
        if op in OPS:
            children = [shape(part) for part in parts]
            if op in (ast.Add, ast.Mult):
                children.sort()
            return repr((op.__name__, children))
        if op in {"call", "neg"}:
            return repr((op, *parts[:-1], shape(parts[-1])))
        return repr(tree)

    seen = {}

    def canonical(tree):
        op, *parts = tree
        if op == "name" and parts[0] in initial:
            return ("parameter", seen.setdefault(parts[0], len(seen)))
        if op in OPS:
            children = sorted(parts, key=shape) if op in (ast.Add, ast.Mult) else parts
            return (op.__name__, *(canonical(part) for part in children))
        if op in {"call", "neg"}:
            return (op, *parts[:-1], canonical(parts[-1]))
        return tree

    def evaluate(tree, values):
        op, *parts = tree
        if op == "number":
            return parts[0]
        if op == "name":
            return values[parts[0]]
        if op == "call":
            return FUNCTIONS[parts[0]](evaluate(parts[1], values))
        if op == "neg":
            return -evaluate(parts[0], values)
        return OPS[op](*(evaluate(part, values) for part in parts))

    names = tuple(initial)

    def model(theta, *params):
        values = {"theta": theta, "pi": jnp.pi, **dict(zip(names, params, strict=True))}
        return jnp.broadcast_to(evaluate(tree, values), jnp.shape(theta))

    def estimator(_theta, _response):
        return tuple(float(initial[name]) for name in names)

    grid = jnp.linspace(-2.9, 2.9, 37)
    values = jnp.asarray(estimator(None, None), dtype=float)
    for params in (values, values + 0.071):
        try:
            a, b = np.asarray(model(grid, *params)), np.asarray(model(grid + 2 * jnp.pi, *params))
        except InvalidEquation:
            raise
        except ArithmeticError as exc:
            raise InvalidEquation("nonfinite_expression", "Equation has undefined arithmetic.") from exc
        except (TypeError, ValueError) as exc:
            raise InvalidEquation(
                "invalid_numeric_expression", "Equation has an invalid numeric operation."
            ) from exc
        if np.iscomplexobj(a) or np.iscomplexobj(b):
            raise InvalidEquation("nonreal_expression", "Equation must produce real-valued responses.")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise InvalidEquation(
                "nonfinite_expression", "Equation produces nonfinite values at trial parameters."
            )
        if not np.allclose(a, b, atol=2e-4, rtol=2e-4):
            raise InvalidEquation("nonperiodic_expression", "Equation failed the 2*pi periodicity probe.")
    key = repr(canonical(tree))
    return Equation(text, dict(initial), key, model, estimator, tuple(seen), proposal.get("nesting"))
