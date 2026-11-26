"""Compare existing baseline CSV files without running new evaluations."""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


SCENARIOS = ("daily", "volatile", "shifted_peak")
POLICIES = ("fixed5", "threshold", "personal")
METRICS = ("reward", "cost", "drop_rate", "switches")
REQUIRED_COLUMNS = {"policy", "scenario", "seed", *METRICS}


def load_results(path: Path, scenario: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}")

        rows = []
        for row in reader:
            if row["scenario"] != scenario or row["policy"] not in POLICIES:
                continue
            result = {
                "scenario": row["scenario"],
                "policy": row["policy"],
                "seed": row["seed"],
            }
            for metric in METRICS:
                result[metric] = float(row[metric])
            rows.append(result)
    return rows


def print_comparison(rows: list[dict]) -> None:
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        if not scenario_rows:
            continue

        grouped = {
            policy: [row for row in scenario_rows if row["policy"] == policy]
            for policy in POLICIES
        }
        seed_sets = {
            policy: {row["seed"] for row in policy_rows}
            for policy, policy_rows in grouped.items()
        }
        matched = (
            bool(seed_sets["fixed5"])
            and seed_sets["fixed5"] == seed_sets["threshold"]
            and seed_sets["fixed5"] == seed_sets["personal"]
        )
        print(f"\n{scenario} — seeds communes: {'oui' if matched else 'NON'}")
        print("baseline   episodes   reward mean (std)   cost mean (std)   "
              "drop rate mean (std)   switches mean (std)")

        for policy in POLICIES:
            policy_rows = grouped[policy]
            if not policy_rows:
                print(f"{policy:<10} aucune donnée")
                continue

            parts = [f"{policy:<10} {len(policy_rows):>8}"]
            for metric in METRICS:
                values = [row[metric] for row in policy_rows]
                mean = statistics.fmean(values)
                std = statistics.pstdev(values) if len(values) > 1 else 0.0
                parts.append(f"{mean:.5f} ({std:.5f})")
            print("   ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize existing evaluate.py CSV outputs; writes no files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing baseline_<scenario>.csv files",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
    )
    args = parser.parse_args()

    all_rows = []
    try:
        for scenario in args.scenarios:
            path = args.input_dir / f"baseline_{scenario}.csv"
            all_rows.extend(load_results(path, scenario))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print_comparison(all_rows)


if __name__ == "__main__":
    main()
