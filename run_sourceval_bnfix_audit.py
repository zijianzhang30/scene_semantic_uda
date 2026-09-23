"""Five-epoch causal audit after adopting batch-stat auxiliary BN."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from audit_causal_pairs import compare

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs_sourceval_bnfix_audit_20260923"
QUEUES = {"4": ["B", "A"], "5": ["C", "F"]}

def main():
    OUT.mkdir(exist_ok=False)
    files = [ROOT / name for name in (
        "train_houston_flow_transport.py", "source_val_protocol.py",
        "class_conditional_flow.py")]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    plan = {"seed": 1341, "epochs": 5, "lr_horizon": 100,
            "selection": "source_val_best", "queues": QUEUES, "files": frozen,
            "formal_ag_remains_paused": True}
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2))
    def queue(gpu, arms):
        for arm in arms:
            for p, digest in frozen.items():
                assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
            dest = OUT / arm / "seed_1341"
            dest.mkdir(parents=True)
            cmd = [sys.executable, "-u", str(ROOT / "train_houston_flow_transport.py"),
                   "--ablation", arm, "--seed", "1341", "--epochs", "5",
                   "--lr-horizon", "100", "--selection", "source_val_best",
                   "--out", str(dest)]
            print("START", arm, "GPU", gpu, flush=True)
            with (dest / "train.log").open("w") as log:
                subprocess.run(cmd, cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONHASHSEED="1341",
                             OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"),
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            print("DONE", arm, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue, gpu, arms) for gpu, arms in QUEUES.items()]
        for future in futures:
            future.result()
    gate = compare(OUT, seed=1341, epochs=5)
    (OUT / "audit_report.json").write_text(json.dumps(gate, indent=2))
    assert gate["passed"], gate
    results = {}
    for arm in ["A", "F", "B", "C"]:
        result = json.loads((OUT / arm / "seed_1341/flow_a/results.json").read_text())
        assert result["selection"] == "source_val_best" and result["training_epochs"] == 5
        results[arm] = result
    for left, right in [("A","F"), ("B","C")]:
        assert all(results[left][k] == results[right][k] for k in (
            "epoch", "source_val_accuracy", "oa", "aa", "kappa"))
    (OUT / "screen_results.json").write_text(json.dumps(results, indent=2))
    print("COMPLETE: BN fix: 5-epoch paired hash and source-selection gates passed; formal A-G remains paused.", flush=True)

if __name__ == "__main__":
    main()
