"""Prompt text for generated and catalog probes."""

import json

from edgar_study.frontier import grammar
from edgar_study.frontier.seed import BASE, BASELINE_ID

SINGLE_EXPRESSION = (
    "Each 'equation' is a single expression in theta: no assignments, statements, 'return' or line breaks."
)


def generated_prompt(choice, nodes, references, k, tried=()):
    selected = choice["node"]
    if choice["kind"] == "revisit":
        idea = nodes[selected]
        task = (
            f"Revisit {idea.label}: {idea.text}. Keep the same idea and give new implementations. "
            f"Return only a JSON array with up to {k} equation objects; each distinct equation is another try."
        )
        if tried:
            task += f" Already tried for this idea (do not repeat): {json.dumps(list(tried))}."
    else:
        mode = choice["mode"]
        brief = {
            "single_idea": "Invent one precise change to the selected parent equation.",
            "anomaly_detour": "Invent one precise change addressing a pattern in the residual plots.",
            "combination": "Invent one idea explicitly combining the selected parents' changes.",
            "independent_root": "Invent one independent model family. The plotted baseline is only a reference.",
        }[mode]
        known = [
            {"title": node.label, "mechanism": node.mechanism, "change": node.text}
            for node in nodes.values()
            if node.routes == ((tuple(choice["parents"])),) or (not node.routes and not choice["parents"])
        ]
        task = (
            f"{brief} Existing ideas on this route (avoid repeats): {json.dumps(known)}. "
            "Return only a JSON object with 'idea' and 'equations'. "
            "The idea has a short, concise 'title' you choose and a concrete 'change'. "
            "The optional 'mechanism' names the changed component, for example offset, width, amplitude, "
            "centre, baseline, shape, kernel, or mixture; it defaults to the title. "
            f"Provide up to {k} structurally distinct equations in 'equations'; each distinct equation is"
            " another try at the idea."
        )
    parents = [
        f"{name}: {BASE}"
        if name == BASELINE_ID
        else f"{name}: {nodes[name].label} — {nodes[name].text}. Equation: {references[name].text}"
        for name in choice["parents"]
    ]
    return (
        f"{task}\nSelected parents: {json.dumps(parents)}.\n"
        "Use theta in radians, periodic with period 2*pi. Use + - * / **, "
        "sin, cos, exp, tanh, abs, sqrt, wrap, or a smooth cutoff using <, <=, >, or >=. "
        f"{SINGLE_EXPRESSION} "
        "Declare every symbol you use except theta and pi in 'initial'. "
        "Each equation object has 'equation', 'initial' (parameter names to numeric starts), and "
        "'parameters' (definitions). Optionally include 'nesting': parent ID to a map giving every parameter "
        "of the new equation as a parent parameter name, a number, or [parent parameter name, scale, shift], "
        "so that the map turns the new equation back into the parent exactly. "
        "For double_gaussian use parent parameter names m, b, a1, a2, s. Do not fit or rank."
    )


def idea_prompt(idea, parents, definitions, references, k):
    if parents:
        lines = [f"- Idea 1 — Base equation: {BASE}"]
        for parent in parents:
            lines.append(
                f"- Idea {len(lines) + 1} — {parent}: {definitions[parent].text}"
                f" Equation: {references[parent].text}"
            )
        lines.append(f"- Idea {len(lines) + 1} — {idea.text}")
        brief = (
            "Combine the ideas in this thread:\n"
            + "\n".join(lines)
            + f"\nGive {k} structurally distinct equations that combine these ideas"
            " and could fit the data shown in the attached images."
        )
    elif idea.lane == "root":
        brief = (
            f"New standalone idea: {idea.text}\nGive {k} structurally distinct full equations based on"
            " this idea that could fit the attached data. The plotted fit is a reference only."
        )
    elif idea.lane == "detour":
        brief = (
            f"Idea to try: {idea.text}\nGive {k} structurally distinct full equations that express this"
            " idea and could fit the attached data. You may change equation family."
        )
    else:
        brief = (
            f"Base equation: {BASE}\nIdea to test: {idea.text}\nGive {k} structurally distinct full"
            " equations that test this idea, keeping other terms unchanged."
        )
    return (
        brief + "\nDo not add an unrelated mechanism. Respect the stated fixed parts. "
        "Use theta in radians, periodic with period 2*pi. Use + - * / ** and "
        "sin, cos, exp, tanh, abs, sqrt or wrap. Comparisons with <, <=, >, or >= give smooth cutoffs."
        f" {SINGLE_EXPRESSION}"
        " Declare every symbol except theta and pi in initial."
        " Keep expressions short. Return only a JSON array of objects with "
        "'equation' (expression), 'initial' (parameter names mapped to numeric starting values), "
        "and 'parameters' (free parameter definitions). If a simpler equation is an exact special case,"
        " optionally include 'nesting': an object keyed by the parent idea ID (or 'double_gaussian' for the base)."
        " Each value maps every parameter of the new equation to a parent parameter name, a numeric constant,"
        " or [parent parameter name, scale, shift] meaning scale * parent + shift. Do not fit or rank."
    )


def tool_note(patience):
    """How the model can check equations before answering, and how many follow-up turns it has."""
    if patience < 1:
        return ""
    return (
        '\nYou may check equations before answering: reply with only {"check": [equation objects]} and'
        " you get a validity report covering syntax, repeats of earlier equations, values at your starting"
        " values, and whether each nesting map turns the equation back into its parent. Equations that"
        f" cannot be used are reported back for replacement. You have {patience} follow-up turns in this"
        " conversation for checks and fixes."
    )


EXAMPLE = {
    "equation": "b + a1*exp(-0.5*(wrap(theta-m)/s)**2) + a2*exp(-0.5*(wrap(theta-m-pi)/s)**2)",
    "initial": {"b": 0.5, "a1": 8.0, "a2": 3.0, "m": 1.0, "s": 0.4},
    "parameters": {
        "b": "baseline",
        "a1": "preferred amplitude",
        "a2": "anti-preferred amplitude",
        "m": "preferred angle",
        "s": "width",
    },
    "nesting": {BASELINE_ID: {"b": "b", "a1": "a1", "a2": "a2", "m": "m", "s": "s"}},
}

_MODES = {
    "single_idea": "Invent one precise change to the selected parent equation.",
    "anomaly_detour": (
        "Invent one precise change that addresses a pattern in the residual plots, especially the "
        "population residual."
    ),
    "combination": (
        "Invent one idea that applies both selected parents' changes together in one equation, so that it "
        "contains each parent as a special case."
    ),
    "independent_root": (
        "Invent one independent model family, not a modified double Gaussian. The plotted fit is only a "
        "reference."
    ),
}

_STATUS = {"fitted": "fitted", "fit_failed": "could not be fitted"}


def _context(scale):
    text = (
        "## Context\n"
        "You are designing a tuning-curve equation for neurons in mouse visual cortex responding to drifting "
        "gratings moving in direction theta (radians, periodic with period 2*pi). One equation is fitted "
        "separately to each of about 800 neurons, so each neuron gets its own parameter values. Many neurons "
        "are weakly tuned or nearly flat. Equations are judged by how well they predict held-out trials of "
        "every neuron, and every parameter has a cost: add a parameter only when it captures a pattern "
        "shared by many neurons."
    )
    if scale:
        text += (
            f" Responses are normalised: a typical neuron's lowest responses are near {scale['baseline']:.2g}"
            f" and its strongest near {scale['peak']:.3g}. Your 'initial' values are the starting point for"
            " every neuron."
        )
    return text


def _rules(k):
    return (
        "## Rules\n"
        f"- {SINGLE_EXPRESSION}\n"
        "- Use + - * / **, sin, cos, exp, tanh, abs, sqrt, wrap, or one smooth cutoff using <, <=, > or >=. "
        "No other functions.\n"
        f"- Use at most {grammar.MAX_PARAMETERS} parameters, at most {grammar.MAX_CHARACTERS} characters; every "
        f"number and starting value between -{grammar.NUMBER_LIMIT} and {grammar.NUMBER_LIMIT}.\n"
        "- Declare every symbol you use except theta and pi in 'initial'.\n"
        "- Keep fits stable for every neuron: never divide by an expression that can cross zero; keep widths "
        "positive (for example s*exp(u) or abs(s)+0.05); bound shape changes with tanh; never let an exponent "
        "grow without limit as a width shrinks (such as k*x/s inside exp), which makes narrow spikes.\n"
        "- Each equation should contain its parent as a special case: give a 'nesting' map from the parent ID "
        "to every parameter of the new equation, as a parent parameter name, a number, or "
        "[parent parameter name, scale, shift], so that the map turns the new equation back into the parent "
        f"exactly. For {BASELINE_ID} use parent parameter names m, b, a1, a2, s.\n"
        f"- Give up to {k} structurally distinct equations; each is another try at the same idea. Do not "
        "fit or rank them."
    )


_IMAGES = (
    "## Images\n"
    "For each parent: one population image, the parent's residual (observed minus fitted) averaged over all "
    "exploration neurons with each neuron's preferred direction at 0 degrees; then 10 neurons, each with its "
    "held-out trials and fit, then its residuals by angle. Look for patterns shared across neurons, mostly in "
    "the population image: peak sharpness, width, skew, preferred and anti-preferred amplitude, the baseline "
    "between peaks. A single noisy neuron is not a pattern."
)


def generated_prompt_v2(choice, nodes, references, k, tried=(), scale=None):
    """The structured prompt: context, task, parents, images, rules, output; never any fit quality."""
    selected = choice["node"]
    if choice["kind"] == "revisit":
        idea = nodes[selected]
        task = f"## Task\nRevisit the idea '{idea.label}': {idea.text} Keep the same idea and give new equations."
        if tried:
            lines = "\n".join(f"- {equation} ({_STATUS.get(status, status)})" for equation, status in tried)
            task += f"\nAlready tried for this idea (do not repeat):\n{lines}"
        output = (
            f"## Output\nReturn only a JSON array of up to {k} equation objects, for example:\n"
            f"{json.dumps([EXAMPLE])}"
        )
    else:
        known = [
            {"title": node.label, "change": node.text}
            for node in nodes.values()
            if node.routes == ((tuple(choice["parents"])),) or (not node.routes and not choice["parents"])
        ]
        task = f"## Task\n{_MODES[choice['mode']]}"
        if known:
            task += f"\nExisting ideas on this route (choose something different): {json.dumps(known)}"
        output = (
            "## Output\nReturn only a JSON object with 'analysis' (at most 600 characters: what the images "
            "show and how your change addresses it), 'idea' ({'title': short, 'mechanism': the changed "
            "component such as width, skew, baseline or amplitude, 'change': concrete description}) and "
            f"'equations' (up to {k} equation objects). An equation object looks like:\n{json.dumps(EXAMPLE)}"
        )
    parents = [
        f"- {name}: {BASE}"
        if name == BASELINE_ID
        else f"- {name} ({nodes[name].label}: {nodes[name].text}): {references[name].text}"
        for name in choice["parents"]
    ]
    sections = [_context(scale), task]
    sections.append("## Parents\n" + ("\n".join(parents) if parents else "None: start a new family."))
    sections += [_IMAGES, _rules(k), output]
    return "\n\n".join(sections)


def _generated_prompt_v1(choice, nodes, references, k, tried=(), scale=None):
    return generated_prompt(choice, nodes, references, k, [equation for equation, _ in tried])


PROMPTS = {"v1": _generated_prompt_v1, "v2": generated_prompt_v2}
