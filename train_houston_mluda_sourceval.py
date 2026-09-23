"""Original full MLUDA objective with the shared source-validation-best evaluator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
from train_houston_flow_transport import (
    LEGACY, ROOT, cfg, UtilsCMS, sha, install_deterministic_pool, parameter_hash)
from source_val_protocol import (
    validation_data, evaluate_source, evaluate_target, isolated_evaluation,
    atomic_save, improves_source_validation)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--lr-horizon",type=int,default=100)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    assert 1 <= args.epochs <= args.lr_horizon
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    out=args.out.resolve()/"official"
    out.mkdir(parents=True,exist_ok=False)
    cache_path=ROOT/"runs_strict_mluda_1341/ilda.npz"
    cache=np.load(cache_path)
    def paired_ilda(s,t,n,r):
        assert n==2 and r==.009
        return cache["s"].copy(),cache["t"].copy()
    UtilsCMS.ILDA=paired_ilda
    cfg.seeds,cfg.nDataSet,cfg.epochs=[args.seed],1,args.epochs
    original=(LEGACY/"MLUDA_hu.py").read_text()
    code=original.split("#################classification map")[0]
    def replace(old,new):
        nonlocal code
        assert code.count(old)==1,old
        code=code.replace(old,new,1)
    replace('feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()',
            'feature_encoder = install_deterministic_pool(DSANSS(nBand, patch_size, CLASS_NUM).cuda())')
    replace('    print("Training...")',
            '    audit_split(trainX, trainY, testX, testY, feature_encoder)\n    print("Training...")')
    replace('        train_end = time.time()',
            '        audit_epoch(epoch, feature_encoder)\n        train_end = time.time()')
    # Both horizons are 100 in formal runs; the option supports a one-epoch smoke test.
    replace('(epoch - 1) / epochs','(epoch - 1) / lr_horizon')
    replace('(epoch) / epochs','(epoch) / lr_horizon')
    def objective(text):
        return text[text.index("            # 0\n"):text.index("            # Update parameters")]
    expected=objective(original).replace('(epoch) / epochs','(epoch) / lr_horizon')
    assert objective(code)==expected
    history=[]
    validation={}
    selected=None
    def audit_split(x,y,tx,ty,model):
        loader,reference,split=validation_data(cache["s"],namespace["label_s"],args.seed,x,y)
        validation.update(loader=loader,reference=reference)
        np.savez(out/"source_validation_split.npz",**split)
        (out/"split.json").write_text(json.dumps(dict(
            seed=args.seed,source_x=sha(x),source_y=sha(y),target_x=sha(tx),target_y=sha(ty),
            source_n=len(y),target_n=len(ty),initial_model=parameter_hash(model)),indent=2))
        (out/"selection_protocol.json").write_text(json.dumps(dict(
            selection="source_val_best",metric="source validation OA",
            tie_rule="earliest epoch",source_forward="model(x,x)[3]",
            source_train_n=len(y),source_validation_n=len(loader.dataset),
            source_validation_centers_hash=sha(split["validation_centers"]),
            source_validation_labels_hash=sha(split["validation_labels"]),
            target_forward="model(train_x[:32], target_batch)[8]",
            target_order="row-major labelled centers",target_drop_last=False,
            target_evaluated_n=int((namespace["label_t"]>0).sum()),
            target_metrics_used_for_selection=False,
            training_protocol="original MLUDA full objective and labelled-mask target sampler",
            pooling="same deterministic backward adapter as A-G",
            objective="CE + 0.01*lambd*LMMD + source SCL + target pseudo-label SCL + occupancy",
            lr_horizon=args.lr_horizon),indent=2))
    def audit_epoch(epoch,model):
        nonlocal selected
        with isolated_evaluation(model):
            val_loss,val_acc=evaluate_source(model,validation["loader"],"cuda")
            target=evaluate_target(model,cache["t"],namespace["label_t"],validation["reference"],"cuda")
        row=dict(epoch=epoch,source_val_loss=val_loss,source_val_accuracy=val_acc,
                 **target,evaluated_n=target["num_target_evaluation_samples"])
        history.append(row)
        checkpoint=dict(model=model.state_dict(),optimizer=namespace["optimizer"].state_dict(),
                        metrics=row,epoch=epoch,selection="source_val_best",
                        seed=args.seed,method="MLUDA_full",
                        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                        numpy_rng=np.random.get_state(),python_rng=random.getstate())
        if improves_source_validation(row,selected):
            selected=dict(row)
            atomic_save(checkpoint,out/"best_source_val.pth")
            (out/"best_source_val.json").write_text(json.dumps(selected,indent=2))
        atomic_save(checkpoint,out/"last.pth")
        (out/"history.json").write_text(json.dumps(history,indent=2))
        print("SOURCEVAL_EPOCH",json.dumps(row),flush=True)
    namespace=dict(__name__="__main__",audit_split=audit_split,audit_epoch=audit_epoch,
                   install_deterministic_pool=install_deterministic_pool,lr_horizon=args.lr_horizon)
    snapshot=out/"code_snapshot";snapshot.mkdir()
    files=[Path(__file__),ROOT/"source_val_protocol.py",ROOT/"train_houston_flow_transport.py"]
    files += [LEGACY/name for name in ["MLUDA_hu.py","net2.py","utils.py","UtilsCMS.py",
                                      "config_Houston.py","mmd.py","contrastive_loss.py"]]
    provenance={}
    for path in files:
        data=path.read_bytes();(snapshot/path.name).write_bytes(data)
        provenance[str(path)]=hashlib.sha256(data).hexdigest()
    (out/"provenance.json").write_text(json.dumps(dict(
        files=provenance,command=sys.argv,torch=torch.__version__,cuda=torch.version.cuda,
        cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        full_objective_preserved=True,
        objective_sha256=hashlib.sha256(expected.encode()).hexdigest()),indent=2))
    (out/"executed.py").write_text(code)
    os.chdir(LEGACY)
    exec(compile(code,str(LEGACY/"MLUDA_hu.py"),"exec"),namespace)
    diagnostic=max(history,key=lambda row:row["oa"])
    result=dict(selected,selection="source_val_best",training_epochs=args.epochs,
                lr_horizon=args.lr_horizon,seed=args.seed,method="MLUDA_full",
                selected_checkpoint="best_source_val.pth",best_epoch=selected["epoch"],
                diagnostic_best_oa=diagnostic["oa"],diagnostic_best_epoch=diagnostic["epoch"],
                fixed_epoch_metrics=dict(history[-1],selection=f"fixed_epoch{args.epochs}"))
    atomic_save(dict(model=namespace["feature_encoder"].state_dict(),
                     metrics=result["fixed_epoch_metrics"]),out/f"epoch{args.epochs}.pth")
    (out/"results.json").write_text(json.dumps(result,indent=2))
    print("FINAL",json.dumps(result),flush=True)

if __name__=="__main__":
    main()
