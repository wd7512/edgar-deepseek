"""Render the JSON lines from the backend scripts as Markdown tables."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def order(row):
    return row["seconds"]


def table(rows, program):
    subset = [row for row in rows if row["program"] == program and row["backend"] != "scipy"]
    subset.sort(key=order)
    lines = [
        "| Rank | Backend | Device | Algorithm | Jacobian | Seconds | Iterations | Train loss | Score |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(subset, 1):
        lines.append(
            f"| {rank} | {row['backend']} | {row['device']} | {row['algorithm']} | "
            f"{row.get('jacobian', '')} | {row['seconds']:.3f} | {row['iterations']} | "
            f"{row['train_loss']:.4f} | {row['score']:.4f} |"
        )
    return "\n".join(lines)


def scipy_table(rows, program):
    subset = [row for row in rows if row["program"] == program and row["backend"] == "scipy"]
    if not subset:
        return ""
    lines = [
        "| Method | Cells | Seconds (subset) | Seconds / 1,055 cells | Evaluations | Train loss | Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in subset:
        lines.append(
            f"| {row['algorithm']} | {row.get('cells', '')} | {row['seconds']:.2f} | "
            f"{row['seconds_per_1055_cells']:.2f} | {row['iterations']} | "
            f"{row['train_loss']:.4f} | {row['score']:.4f} |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=ROOT / "scripts/backend_bench/results.jsonl")
    args = parser.parse_args(argv)
    rows = read(args.input)
    programs = []
    for row in rows:
        if row["program"] not in programs:
            programs.append(row["program"])
    for program in programs:
        print(f"### {program}\n")
        print(table(rows, program))
        scipy = scipy_table(rows, program)
        if scipy:
            print("\nSciPy, single-threaded on a 200-cell subset:\n")
            print(scipy)
        print()


if __name__ == "__main__":
    main()
