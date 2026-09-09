"""Post-hoc evaluation for sample-wise Transferability-Aware Scene Shift."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import hdf5storage, numpy as np, torch
from sklearn import metrics
import train as clean
from train_samplewise_transferability import PatchDomainDiscriminator, summarize
import utils
from model import DCRNClassifier

ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); HERE=Path(__file__).resolve().parent

def patches(cube, centers, width=7):
    h=width//2; p=np.pad(cube,((h,h),(h,h),(0,0)),mode='constant'); out=np.empty((len(centers),cube.shape[-1],width,width),np.float32)
    for i,(r,c) in enumerate(centers): out[i]=p[r:r+width,c:c+width].transpose(2,0,1)
    return out

def logits(model,x,device,batch=256):
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),batch): out.append(model(torch.from_numpy(x[i:i+batch]).to(device)).cpu())
    return torch.cat(out)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=HERE/'runs_samplewise_transferability_1174'); p.add_argument('--sweep-root',type=Path,default=HERE/'runs_scene_shift_strength_sweep_1174'); p.add_argument('--diagnostic-root',type=Path,default=HERE/'runs_transferability_aware_1174'); p.add_argument('--device',default='cuda:0'); args=p.parse_args(); device=torch.device(args.device)
    source,source_gt=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat')); target=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']; target_gt=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map']; source=source.astype(np.float32); target=target.astype(np.float32)
    _,_,vc,vy=clean.source_split(source_gt,1174); val_x=clean.center_patches(source,vc); centers=np.argwhere(target_gt>0).astype(np.int64); y=target_gt[centers[:,0],centers[:,1]].astype(np.int64)-1; target_eval_x=patches(target,centers)
    ck=torch.load(args.root/'best.pth',map_location='cpu'); model=DCRNClassifier().to(device); model.load_state_dict(ck['model']); model.eval(); pred=logits(model,target_eval_x,device).argmax(1).numpy(); cm=metrics.confusion_matrix(y,pred,labels=np.arange(7)); pc=np.diag(cm)/np.maximum(cm.sum(1),1)
    results={'DCRN + CE (alpha=0)':None,'Global Scene Shift alpha=0.8':None,'Sample-wise Transferability Scene Shift':{'oa':float((y==pred).mean()),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,pred,labels=np.arange(7))),'per_class_accuracy':pc.tolist(),'best_epoch':ck['best']['epoch'],'source_val_accuracy':ck['best']['val_acc']}}
    sweep=json.loads((args.sweep_root/'summary.json').read_text())['runs']; results['DCRN + CE (alpha=0)']=next({k:r[k] for k in ('oa','aa','kappa','per_class_accuracy','best_epoch','source_val_accuracy')} for r in sweep if r['alpha']==0.0); results['Global Scene Shift alpha=0.8']=next({k:r[k] for k in ('oa','aa','kappa','per_class_accuracy','best_epoch','source_val_accuracy')} for r in sweep if r['alpha']==0.8)
    with (args.root/'per_class_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['model','class','accuracy']);
        for name,r in results.items():
            for c,a in enumerate(r['per_class_accuracy'],1): w.writerow([name,c,a])
    # Distribution and per-class stats at the saved source-val-best model/discriminator.
    discriminator=PatchDomainDiscriminator(48).to(device); discriminator.load_state_dict(ck['discriminator']); discriminator.eval(); sf,tf=source.reshape(-1,48),target.reshape(-1,48); sm,ss=sf.mean(0),sf.std(0); tm,ts=tf.mean(0),tf.std(0); tc,ty,_,_=clean.source_split(source_gt,1174); train_x=clean.center_patches(source,tc)
    torch.manual_seed(20260909); shifted=[]
    with torch.no_grad():
        for i in range(0,len(train_x),256): shifted.append(clean.scene_shift(torch.from_numpy(train_x[i:i+256]).to(device),sm,ss,tm,ts,strength=0.8).cpu())
    shifted_x=torch.cat(shifted).numpy(); raw_logits=logits(model,train_x,device); shift_logits=logits(model,shifted_x,device); labels_t=torch.from_numpy(ty)
    with torch.no_grad(): affinity=torch.sigmoid(discriminator(torch.from_numpy(shifted_x).to(device))).cpu().numpy()
    raw_margin=(raw_logits.gather(1,labels_t[:,None]).squeeze(1)-raw_logits.masked_fill(torch.nn.functional.one_hot(labels_t,7).bool(),-torch.inf).max(1).values).numpy(); shift_margin=(shift_logits.gather(1,labels_t[:,None]).squeeze(1)-shift_logits.masked_fill(torch.nn.functional.one_hot(labels_t,7).bool(),-torch.inf).max(1).values).numpy(); reliability=np.exp(-np.maximum(raw_margin-shift_margin,0)); weight=np.clip(affinity*reliability,0.2,1.0)
    class_rec={}
    for c in range(7):
        m=ty==c; class_rec[str(c+1)]={'affinity':summarize(affinity[m]),'reliability':summarize(reliability[m]),'weight':summarize(weight[m])}
    transfer={'source_train_best_checkpoint':True,'alpha':0.8,'affinity':summarize(affinity),'reliability':summarize(reliability),'weight':summarize(weight),'per_class':class_rec,'target_gt_used_for_training_or_selection':False}
    (args.root/'transferability_stats.json').write_text(json.dumps(transfer,indent=2))
    # Domain discriminator accuracy and source/target/shifted affinity means.
    src_prob=[]; tgt_prob=[]
    with torch.no_grad():
        for i in range(0,len(train_x),256): src_prob.append(torch.sigmoid(discriminator(torch.from_numpy(train_x[i:i+256]).to(device))).cpu().numpy())
        for i in range(0,len(target_eval_x),256): tgt_prob.append(torch.sigmoid(discriminator(torch.from_numpy(target_eval_x[i:i+256]).to(device))).cpu().numpy())
    src_prob=np.concatenate(src_prob); tgt_prob=np.concatenate(tgt_prob); disc={'source_raw_target_prob':summarize(src_prob),'target_raw_target_prob':summarize(tgt_prob),'shifted_source_target_prob':summarize(affinity),'source_accuracy':float((src_prob<0.5).mean()),'target_accuracy':float((tgt_prob>=0.5).mean()),'balanced_accuracy':float(0.5*((src_prob<0.5).mean()+(tgt_prob>=0.5).mean())),'target_gt_used':False}; (args.root/'discriminator_stats.json').write_text(json.dumps(disc,indent=2))
    out={'protocol':{'split':1174,'target_gt':'post-hoc evaluation only','alpha':0.8,'classifier_loss':'mean(CE_raw_i + w_i CE_shift_i)','discriminator_loss':'BCE(source,0)+BCE(target,1)','target_gt_used_for_training_or_selection':False},'results':results,'transferability_stats':transfer,'discriminator_stats':disc}; (args.root/'result.json').write_text(json.dumps(out,indent=2))
    # Summary markdown with the requested diagnosis.
    g=results['Global Scene Shift alpha=0.8']; s=results['Sample-wise Transferability Scene Shift']; c6=(s['per_class_accuracy'][5]-g['per_class_accuracy'][5])*100; c3=(s['per_class_accuracy'][2]-g['per_class_accuracy'][2])*100; c7=(s['per_class_accuracy'][6]-g['per_class_accuracy'][6])*100
    md=f'''# Sample-wise Transferability-Aware Scene Shift (split 1174)\n\nTraining uses fixed global Scene Shift alpha=0.8, independent patch discriminator, and source-validation-only checkpoint selection. Target GT is used only in this post-hoc evaluation.\n\n|Method|OA|AA|Kappa|Best epoch|\n|---|---:|---:|---:|---:|\n|DCRN + CE|{results["DCRN + CE (alpha=0)"]["oa"]*100:.2f}|{results["DCRN + CE (alpha=0)"]["aa"]*100:.2f}|{results["DCRN + CE (alpha=0)"]["kappa"]*100:.2f}|{results["DCRN + CE (alpha=0)"]["best_epoch"]}|\n|Global Scene Shift alpha=0.8|{g["oa"]*100:.2f}|{g["aa"]*100:.2f}|{g["kappa"]*100:.2f}|{g["best_epoch"]}|\n|Sample-wise reweight|{s["oa"]*100:.2f}|{s["aa"]*100:.2f}|{s["kappa"]*100:.2f}|{s["best_epoch"]}|\n\nClass-wise sample weights (C1–C7): `{json.dumps(ck.get("best",{}).get("class_stats",{}))}`\n\nCompared with global alpha=0.8: C6 {c6:+.2f} pp, C3 {c3:+.2f} pp, C7 {c7:+.2f} pp.\n\nDiscriminator: source accuracy {disc["source_accuracy"]*100:.2f}%, target accuracy {disc["target_accuracy"]*100:.2f}%, balanced accuracy {disc["balanced_accuracy"]*100:.2f}%. Mean target probability: raw source {disc["source_raw_target_prob"]["mean"]:.3f}, shifted source {disc["shifted_source_target_prob"]["mean"]:.3f}, raw target {disc["target_raw_target_prob"]["mean"]:.3f}.\n\nConclusion: inspect `transferability_stats.json` before extending. This single split is not sufficient to claim an overall gain; no hyperparameter sweep or 3-seed extension is implied.\n'''; (args.root/'summary.md').write_text(md); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
