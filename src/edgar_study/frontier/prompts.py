"""Prompt text for generated and catalog probes."""

import json

from edgar_study.frontier.seed import BASE, BASELINE_ID


def generated_prompt(choice, nodes, references, k, tried=()):
    selected = choice["node"]
    if choice["kind"] == "revisit":
        idea = nodes[selected]
        task = (
            f"Revisit {idea.label}: {idea.text}. Keep the same idea and give new implementations. "
            f"Return only a JSON array with zero to {k} equation objects."
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
            f"Provide zero to {k} structurally distinct equations in 'equations'."
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
        "Declare every symbol you use except theta and pi in 'initial'. "
        "Each equation object has 'equation', 'initial' (parameter names to numeric starts), and "
        "'parameters' (definitions). Optionally include 'nesting': parent ID to a map from each new "
        "parameter to a parent parameter name, a number, or [parent parameter name, scale, shift]. "
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
        " Declare every symbol except theta and pi in initial."
        " Keep expressions short. Return only a JSON array of objects with "
        "'equation' (expression), 'initial' (parameter names mapped to numeric starting values), "
        "and 'parameters' (free parameter definitions). If a simpler equation is an exact special case,"
        " optionally include 'nesting': an object keyed by the parent idea ID (or 'double_gaussian' for the base)."
        " Each value maps every new equation parameter to a parent parameter name, a numeric constant, or"
        " [parent parameter name, scale, shift] meaning scale * parent + shift. Do not fit or rank."
    )
