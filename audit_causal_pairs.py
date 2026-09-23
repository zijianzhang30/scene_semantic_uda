"""Fail closed unless paired real-training trajectories agree bit for bit."""
import argparse
import json
from pathlib import Path


def compare(root, seed=1341, epochs=5):
    report = {"seed": seed, "epochs": epochs, "pairs": {}, "passed": True}
    for left, right in [("A", "F"), ("B", "C")]:
        dirs = [root / a / f"seed_{seed}" / "flow_a" for a in (left, right)]
        rows = [[json.loads(s) for s in (p / "step_audit.jsonl").read_text().splitlines()] for p in dirs]
        splits = [json.loads((p / "split.json").read_text()) for p in dirs]
        histories = [json.loads((p / "history.json").read_text()) for p in dirs]
        provenances = [json.loads((p / "provenance.json").read_text()) for p in dirs]
        expected = [(epoch, step) for epoch in range(1, epochs + 1) for step in range(1, 39)]
        complete = all([(r["epoch"], r["step"]) for r in seq] == expected for seq in rows)
        complete &= all([r["epoch"] for r in hist] == list(range(1, epochs + 1)) for hist in histories)
        same_start = all(splits[0][k] == splits[1][k] for k in (
            "source_x", "source_y", "target_x", "target_y", "initial_model", "official_file_sha256"))
        same_code = provenances[0]["files"] == provenances[1]["files"]
        differences = [{"epoch": a["epoch"], "step": a["step"],
                        "fields": [k for k in set(a) | set(b) if a.get(k) != b.get(k)]}
                       for a, b in zip(*rows) if a != b]
        evaluation_equal = all(all(a[k] == b[k] for k in (
            "oa", "aa", "kappa", "per_class_accuracy")) for a, b in zip(*histories))
        selection_equal = True
        if any("source_val_accuracy" in h[0] for h in histories):
            selection_equal = all(all(a.get(k) == b.get(k) for k in (
                "source_val_accuracy", "source_val_loss")) for a, b in zip(*histories))
            selected = [max(h, key=lambda row: row["source_val_accuracy"]) for h in histories]
            selection_equal &= selected[0]["epoch"] == selected[1]["epoch"]
        passed = complete and same_start and same_code and not differences and evaluation_equal and selection_equal
        report["pairs"][left + "/" + right] = {
            "passed": passed, "steps": [len(r) for r in rows], "complete": complete,
            "same_initialization_and_split": same_start, "same_code": same_code,
            "evaluation_equal": evaluation_equal, "source_selection_equal": selection_equal,
            "mismatch_steps": len(differences), "first_mismatch": differences[:1],
        }
        report["passed"] &= passed
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", type=int, default=1341)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    result = compare(args.root, args.seed, args.epochs)
    (args.root / "audit_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
