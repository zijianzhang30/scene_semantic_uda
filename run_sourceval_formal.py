"""Run source-val-best A-G after a matching five-epoch causal audit."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from audit_causal_pairs import compare

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--baseline-smoke",type=Path,required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arms", nargs="+", choices=["MLUDA"]+list("ABCDEFG"), required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[1341,1174,1370])
    p.add_argument("--gpus", nargs="+", default=["1","4","5"])
    args = p.parse_args()
    assert len(args.seeds) == len(args.gpus)
    assert len(set(args.seeds)) == len(args.seeds)
    assert len(set(args.gpus)) == len(args.gpus)
    audit, out = args.audit.resolve(), args.out.resolve()
    gate = compare(audit, seed=1341, epochs=5)
    assert gate["passed"], gate
    provenance = json.loads((audit/"B/seed_1341/flow_a/provenance.json").read_text())
    protocol = json.loads((audit/"B/seed_1341/flow_a/selection_protocol.json").read_text())
    assert protocol["selection"] == "source_val_best"
    assert protocol["protocol_version"] == "sourceval_bn_batch_restore_v2"
    smoke=args.baseline_smoke.resolve()/"official"
    smoke_result=json.loads((smoke/"results.json").read_text())
    assert smoke_result["selection"]=="source_val_best" and smoke_result["training_epochs"]==1
    baseline_provenance=json.loads((smoke/"provenance.json").read_text())
    assert baseline_provenance["full_objective_preserved"]
    for name,digest in baseline_provenance["files"].items():
        if name in provenance["files"]: assert provenance["files"][name]==digest,name
        provenance["files"][name]=digest
    def verify():
        for name, digest in provenance["files"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    verify()
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(arms=args.arms, seeds=args.seeds, gpus=args.gpus, epochs=100,
                lr_horizon=100, selection="source_val_best",
                protocol=protocol, provenance=provenance, audit=str(audit))
    (out/"plan.json").write_text(json.dumps(plan, indent=2))
    (out/"five_epoch_gate.json").write_text(json.dumps(gate, indent=2))
    lock = threading.Lock()
    failed = threading.Event()
    states = {seed: dict(gpu=gpu, arm=None, status="queued")
              for seed,gpu in zip(args.seeds,args.gpus)}
    rows = []
    def update():
        (out/"status.json").write_text(json.dumps(dict(workers=states,failed=failed.is_set()),indent=2))
        if rows:
            with (out/"source_val_best.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
    def queue(seed,gpu):
        try:
            for arm in args.arms:
                if failed.is_set(): return
                verify()
                dest=out/arm/f"seed_{seed}"
                dest.mkdir(parents=True,exist_ok=False)
                if arm=="MLUDA":
                    cmd=[sys.executable,"-u",str(ROOT/"train_houston_mluda_sourceval.py"),
                         "--seed",str(seed),"--epochs","100","--lr-horizon","100","--out",str(dest)]
                    result_dir=dest/"official"
                else:
                    cmd=[sys.executable,"-u",str(ROOT/"train_houston_flow_transport.py"),
                         "--ablation",arm,"--seed",str(seed),"--epochs","100",
                         "--lr-horizon","100","--selection","source_val_best","--out",str(dest)]
                    result_dir=dest/"flow_a"
                with lock:
                    states[seed].update(arm=arm,status="running")
                    update()
                    print("START",arm,seed,"GPU",gpu,flush=True)
                with (dest/"train.log").open("w") as log:
                    subprocess.run(cmd,cwd=ROOT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED=str(seed),
                                 OMP_NUM_THREADS="2",MKL_NUM_THREADS="2"),
                        stdout=log,stderr=subprocess.STDOUT,check=True)
                r=json.loads((result_dir/"results.json").read_text())
                assert r["selection"]=="source_val_best" and r["training_epochs"]==100
                row=dict(arm=arm,seed=seed,selected_epoch=r["epoch"],
                         source_val_OA=100*r["source_val_accuracy"],
                         OA=100*r["oa"],AA=100*r["aa"],Kappa=100*r["kappa"],
                         **{f"class_{i+1}":100*v for i,v in enumerate(r["per_class_accuracy"])},
                         epoch100_OA=100*r["fixed_epoch_metrics"]["oa"],
                         target_best_diagnostic_OA=100*r["diagnostic_best_oa"],
                         target_best_diagnostic_epoch=r["diagnostic_best_epoch"])
                if all((out/a/f"seed_{seed}/flow_a/results.json").exists() for a in ["A","F","B","C"]):
                    check=compare(out,seed=seed,epochs=100)
                    (out/f"paired_hash_audit_seed_{seed}.json").write_text(json.dumps(check,indent=2))
                    assert check["passed"],check
                with lock:
                    rows.append(row)
                    states[seed].update(status="completed_arm")
                    update()
                    print("DONE",arm,seed,"source-val selected epoch",r["epoch"],flush=True)
            with lock:
                states[seed].update(status="complete")
                update()
        except Exception as exc:
            failed.set()
            with lock:
                states[seed].update(status="failed",error=repr(exc))
                update()
            raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        tasks=[pool.submit(queue,seed,gpu) for seed,gpu in zip(args.seeds,args.gpus)]
        for task in tasks:task.result()
    print("COMPLETE source-val-best formal runs",flush=True)

if __name__=="__main__":
    main()
