"""Eight GPU queues for full MLUDA and A-G, gated by causal audits."""
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
import time
from audit_causal_pairs import compare

ROOT=Path(__file__).resolve().parent
ARMS=["MLUDA","A","F","B","C","D","E","G"]
SEEDS=[1341,1174,1370]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--baseline-smoke",type=Path,required=True)
    p.add_argument("--gpus",nargs="+",default=[str(i) for i in range(8)])
    p.add_argument("--min-free-mib",type=int,default=3072)
    args=p.parse_args()
    assert len(args.gpus)==len(set(args.gpus))
    out,audit=args.out.resolve(),args.audit.resolve()
    gate=compare(audit,seed=1341,epochs=5)
    assert gate["passed"],gate
    protocol=json.loads((audit/"B/seed_1341/flow_a/selection_protocol.json").read_text())
    assert protocol["protocol_version"]=="sourceval_bn_batch_restore_v2"
    provenance=json.loads((audit/"B/seed_1341/flow_a/provenance.json").read_text())
    smoke=args.baseline_smoke.resolve()/"official"
    result=json.loads((smoke/"results.json").read_text())
    assert result["selection"]=="source_val_best" and result["training_epochs"]==1
    baseline=json.loads((smoke/"provenance.json").read_text())
    assert baseline["full_objective_preserved"]
    for name,digest in baseline["files"].items():
        if name in provenance["files"]: assert provenance["files"][name]==digest,name
        provenance["files"][name]=digest
    def verify():
        for name,digest in provenance["files"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    verify()
    assert out.is_dir()
    (out/"resume_plan.json").write_text(json.dumps(dict(
        arms=ARMS,seeds=SEEDS,gpus=args.gpus,epochs=100,lr_horizon=100,
        selection="source_val_best",protocol=protocol,provenance=provenance,
        min_free_mib=args.min_free_mib,
        scheduling="dynamic GPU queues; D/E/G wait for the same-seed 100-epoch A/F and B/C gate"),indent=2))
    (out/"five_epoch_gate.json").write_text(json.dumps(gate,indent=2))
    lock=threading.Lock()
    failed=threading.Event()
    saved=json.loads((out/"status_before_six_gpu.json").read_text())
    assert not saved["failed"]
    jobs=saved["jobs"]
    assert not any(j["status"]=="complete" for j in jobs), "Use result import before adopting completed jobs"
    assert all(j["gpu"] in args.gpus for j in jobs if j["status"]=="running")
    workers={gpu:dict(status="initializing") for gpu in args.gpus}
    rows=[]
    passed_seeds=set()
    def update():
        temp=out/"status.json.tmp"
        temp.write_text(json.dumps(dict(failed=failed.is_set(),workers=workers,jobs=jobs,
                                       passed_100epoch_seeds=sorted(passed_seeds)),indent=2))
        temp.replace(out/"status.json")
        if rows:
            with (out/"source_val_best.csv.tmp").open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            (out/"source_val_best.csv.tmp").replace(out/"source_val_best.csv")
    def queue(gpu):
        current=None
        try:
            while not failed.is_set():
                with lock:
                    if all(j["status"]=="complete" for j in jobs):
                        workers[gpu]=dict(status="complete");update();return
                disabled_path=out/"disabled_gpus.json"
                disabled=json.loads(disabled_path.read_text()) if disabled_path.exists() else []
                if gpu in disabled:
                    with lock:
                        workers[gpu]=dict(status="disabled_by_user")
                        update()
                    return
                free=int(subprocess.check_output([
                    "nvidia-smi",f"--id={gpu}","--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"],text=True).strip())
                with lock:
                    adopted=[j for j in jobs if j["status"]=="running" and j["gpu"]==gpu]
                    candidates=adopted+[j for j in jobs if j["status"]=="pending" and
                                (j["arm"] not in ["D","E","G"] or j["seed"] in passed_seeds)]
                    if free<args.min_free_mib and not adopted:
                        workers[gpu]=dict(status="waiting_memory",free_mib=free);update()
                    elif not candidates:
                        workers[gpu]=dict(status="waiting_dependencies",free_mib=free);update()
                    else:
                        current=candidates[0];current.update(status="running",gpu=gpu)
                        workers[gpu]=dict(status="running",arm=current["arm"],seed=current["seed"])
                        update()
                if current is None:
                    time.sleep(10);continue
                verify()
                arm,seed=current["arm"],current["seed"]
                dest=out/arm/f"seed_{seed}";dest.mkdir(parents=True,exist_ok=("pid" in current))
                if arm=="MLUDA":
                    cmd=[sys.executable,"-u",str(ROOT/"train_houston_mluda_sourceval.py")]
                    result_dir=dest/"official"
                else:
                    cmd=[sys.executable,"-u",str(ROOT/"train_houston_flow_transport.py"),
                         "--ablation",arm,"--selection","source_val_best"]
                    result_dir=dest/"flow_a"
                cmd+=["--seed",str(seed),"--epochs","100","--lr-horizon","100","--out",str(dest)]
                if "pid" in current:
                    adopted_pid=current["pid"]
                    with lock:
                        workers[gpu]["pid"]=adopted_pid;update()
                        print("ADOPT",arm,seed,"GPU",gpu,"PID",adopted_pid,flush=True)
                    while True:
                        proc=Path(f"/proc/{adopted_pid}")
                        try:
                            state=(proc/"stat").read_text().rsplit(")",1)[1].split()[0]
                            if state=="Z": break
                            running_cmd=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode()
                            assert str(dest) in running_cmd, "Adopted PID identity changed"
                        except FileNotFoundError:
                            break
                        time.sleep(5)
                    assert (result_dir/"results.json").exists(),f"Adopted run failed: {dest}"
                else:
                    with (dest/"train.log").open("w") as log:
                        process=subprocess.Popen(cmd,cwd=ROOT,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED=str(seed),
                                     OMP_NUM_THREADS="2",MKL_NUM_THREADS="2"),
                            stdout=log,stderr=subprocess.STDOUT)
                        with lock:
                            current["pid"]=process.pid;workers[gpu]["pid"]=process.pid;update()
                            print("START",arm,seed,"GPU",gpu,"PID",process.pid,flush=True)
                        assert process.wait()==0,f"{arm} seed {seed}: failed, see {dest/'train.log'}"
                r=json.loads((result_dir/"results.json").read_text())
                assert r["selection"]=="source_val_best" and r["training_epochs"]==100
                row=dict(arm=arm,seed=seed,selected_epoch=r["epoch"],
                         source_val_OA=100*r["source_val_accuracy"],OA=100*r["oa"],
                         AA=100*r["aa"],Kappa=100*r["kappa"],
                         **{f"class_{i+1}":100*v for i,v in enumerate(r["per_class_accuracy"])},
                         epoch100_OA=100*r["fixed_epoch_metrics"]["oa"],
                         target_best_diagnostic_OA=100*r["diagnostic_best_oa"],
                         target_best_diagnostic_epoch=r["diagnostic_best_epoch"])
                with lock:
                    current["status"]="complete";rows.append(row)
                    if seed not in passed_seeds and all(any(
                        j["arm"]==a and j["seed"]==seed and j["status"]=="complete"
                        for j in jobs) for a in ["A","F","B","C"]):
                        check=compare(out,seed=seed,epochs=100)
                        (out/f"paired_hash_audit_seed_{seed}.json").write_text(json.dumps(check,indent=2))
                        assert check["passed"],check
                        passed_seeds.add(seed)
                    print("DONE",arm,seed,"selected epoch",r["epoch"],flush=True)
                    workers[gpu]=dict(status="available");update()
                current=None
        except Exception as exc:
            failed.set()
            with lock:
                if current is not None:current.update(status="failed",error=repr(exc))
                workers[gpu]=dict(status="failed",error=repr(exc));update()
            raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures=[pool.submit(queue,gpu) for gpu in args.gpus]
        for future in futures:future.result()
    assert not failed.is_set()
    print("COMPLETE: all 24 source-val-best runs and all full-trajectory gates passed.",flush=True)

if __name__=="__main__":
    main()
