"""Read-only offline mechanism analysis for matched Houston B/C checkpoints."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.sceneshift_net_dcrn import SceneShiftNetDCRN
from sceneshiftnet_dcrn_train import PatchDataset,file_sha256
from train_sceneshiftnet_dcrn_houston_raw import load_cubes

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs_sceneshiftnet_dcrn/houston_raw/three_seed_final'
OUT=RUN/'mechanism_analysis_B_vs_C'
SEEDS=(1341,2024,3407);METHODS=('B','C');PAIRS=((1,2),(1,6),(1,7),(2,6),(2,5))


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


@torch.no_grad()
def infer(model,loader,device):
    features=[];probabilities=[];truth=[]
    model.eval()
    for x,y in loader:
        f,z=model(x.to(device));features.append(f.cpu().numpy());probabilities.append(z.softmax(1).cpu().numpy());truth.append(y.numpy())
    return np.concatenate(features),np.concatenate(probabilities),np.concatenate(truth)


def matrices(prob,truth):
    pred=prob.argmax(1);hard=np.zeros((7,7),dtype=np.int64)
    np.add.at(hard,(truth,pred),1);normalized=hard/hard.sum(1,keepdims=True)
    soft=np.stack([prob[truth==c].mean(0) for c in range(7)])
    return soft,hard,normalized


def geometry(features,truth):
    centroids=np.stack([features[truth==c].mean(0) for c in range(7)])
    radii=np.array([np.linalg.norm(features[truth==c]-centroids[c],axis=1).mean() for c in range(7)])
    rows=[]
    for a,b in PAIRS[:-1]:
        x,y=centroids[a-1],centroids[b-1];cos=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12));eu=float(np.linalg.norm(x-y));avg_radius=float((radii[a-1]+radii[b-1])/2)
        rows.append(dict(Class_i=a,Class_j=b,centroid_cosine=cos,centroid_euclidean=eu,radius_i=radii[a-1],radius_j=radii[b-1],inter_intra_ratio=eu/(avg_radius+1e-12)))
    return centroids,radii,rows


def summarize_deltas(rows,group_keys,value_keys,out_rows,analysis):
    grouped=defaultdict(list)
    for r in rows:
        if r['Method']=='Delta_C_minus_B':grouped[tuple(r[k] for k in group_keys)].append(r)
    for key,values in grouped.items():
        for metric in value_keys:
            data=np.array([float(v[metric]) for v in values]);out_rows.append({'Analysis':analysis,**dict(zip(group_keys,key)),'Metric':metric,'Mean_Delta_C_minus_B':data.mean(),'Std':data.std(ddof=0),'Positive_Count':int((data>0).sum()),'Negative_Count':int((data<0).sum()),'Seeds':len(data)})


def main():
    OUT.mkdir(parents=True,exist_ok=True);device='cuda' if torch.cuda.is_available() else 'cpu'
    source,sg,target,tg,paths=load_cubes('none');source=source.astype('float32');target=target.astype('float32')
    target_centers=np.argwhere(tg>0);target_labels=tg[target_centers[:,0],target_centers[:,1]].astype('int64')-1
    target_loader=DataLoader(PatchDataset(target,target_centers,7,target_labels),args.batch_size,shuffle=False,num_workers=0)
    sm,ss=source.reshape(-1,48).mean(0),source.reshape(-1,48).std(0);tm,ts=target.reshape(-1,48).mean(0),target.reshape(-1,48).std(0)
    d=(np.abs(tm-sm)+np.abs(ts-ss))/(ss+1e-5);alpha=.4+.5*(d-d.min())/(d.max()-d.min()+1e-5);high=alpha>=np.quantile(alpha,.75)
    confusion_rows=[];mass_rows=[];geometry_rows=[];shift_rows=[];provenance={'script_hash':file_sha256(__file__),'data_hash':{str(p):file_sha256(p) for p in paths},'seeds':SEEDS,'methods':METHODS,'pairs':PAIRS,'target_samples':len(target_centers),'definitions':{'soft_matrix':'mean softmax vector within each GT class','hard_normalized':'row-normalized by GT class','radius':'mean Euclidean feature distance to class centroid','inter_intra':'centroid Euclidean / mean(pair radii)','high_alpha':'top quartile alpha bands','target_gt':'offline grouping only'}}
    all_outputs={}
    for seed in SEEDS:
        saved=np.load(RUN/'B'/f'seed_{seed}'/'indices.npz');centers=saved['source'];labels=saved['source_labels']
        source_ds=PatchDataset(source,centers,7);shifted=(source-sm)/(ss+1e-5)*(alpha*ts+(1-alpha)*ss)+alpha*tm+(1-alpha)*sm;shift_ds=PatchDataset(shifted,centers,7)
        source_loader=DataLoader(source_ds,args.batch_size,shuffle=False);shift_loader=DataLoader(shift_ds,args.batch_size,shuffle=False)
        seed_results={}
        for method in METHODS:
            ck=torch.load(RUN/method/f'seed_{seed}'/'final.pth',map_location=device,weights_only=False);model=SceneShiftNetDCRN(48,7,7).to(device);model.load_state_dict(ck['model'])
            ft,pt,yt=infer(model,target_loader,device);soft,hard,norm=matrices(pt,yt);centroids,radii,grows=geometry(ft,yt)
            seed_results[method]=dict(soft=soft,hard=hard,norm=norm,centroids=centroids,radii=radii,geometry=grows)
            np.savez_compressed(OUT/f'{method}_seed_{seed}_matrices.npz',mean_soft_prediction_matrix=soft,hard_confusion_matrix=hard,normalized_confusion_matrix=norm,class_centroids=centroids,within_class_radius=radii)
            for i,j in PAIRS:
                confusion_rows.append(dict(Seed=seed,Method=method,GT_Class=i,Pred_Class=j,Hard_Count=int(hard[i-1,j-1]),Hard_Conditional_Probability=norm[i-1,j-1],Mean_Soft_Probability=soft[i-1,j-1]))
                confusion_rows.append(dict(Seed=seed,Method=method,GT_Class=j,Pred_Class=i,Hard_Count=int(hard[j-1,i-1]),Hard_Conditional_Probability=norm[j-1,i-1],Mean_Soft_Probability=soft[j-1,i-1]))
            for c in range(7):
                for k in range(7):mass_rows.append(dict(Seed=seed,Method=method,GT_Class=c+1,Probability_Class=k+1,Mean_Probability=soft[c,k]))
            for r in grows:geometry_rows.append({'Seed':seed,'Method':method,**r})
            fs,_,_=infer(model,DataLoader(PatchDataset(source,centers,7,labels),args.batch_size,shuffle=False),device);fsh,_,_=infer(model,DataLoader(PatchDataset(shifted,centers,7,labels),args.batch_size,shuffle=False),device)
            cs=np.stack([fs[labels==c].mean(0) for c in range(7)]);csh=np.stack([fsh[labels==c].mean(0) for c in range(7)])
            rs=np.array([np.linalg.norm(fs[labels==c]-cs[c],axis=1).mean() for c in range(7)]);rsh=np.array([np.linalg.norm(fsh[labels==c]-csh[c],axis=1).mean() for c in range(7)])
            cosmat=lambda c: F.normalize(torch.from_numpy(c),dim=1).numpy()@F.normalize(torch.from_numpy(c),dim=1).numpy().T
            before_cos,after_cos=cosmat(cs),cosmat(csh)
            patches=np.stack([source_ds[i].numpy() for i in range(len(source_ds))]);spatches=np.stack([shift_ds[i].numpy() for i in range(len(shift_ds))]);band_delta=np.abs(spatches-patches).mean((2,3))
            for c in range(7):
                own=float(cs[c]@csh[c]/(np.linalg.norm(cs[c])*np.linalg.norm(csh[c])+1e-12));before=before_cos[c].copy();after=after_cos[c].copy();before[c]=after[c]=-np.inf
                cd=band_delta[labels==c].mean(0);row=dict(Seed=seed,Method=method,Class=c+1,Centroid_Euclidean_Movement=float(np.linalg.norm(csh[c]-cs[c])),Centroid_Cosine_Movement=float(1-own),Within_Class_Radius_Before=rs[c],Within_Class_Radius_After=rsh[c],Spread_Change=rsh[c]-rs[c],Nearest_Class_Before=int(before.argmax()+1),Nearest_Class_After=int(after.argmax()+1),Nearest_Cosine_Before=float(before.max()),Nearest_Cosine_After=float(after.max()),High_Alpha_Mean_Abs_Input_Change=float(cd[high].mean()),Low_Alpha_Mean_Abs_Input_Change=float(cd[~high].mean()),High_Alpha_Change_Fraction=float(cd[high].sum()/(cd.sum()+1e-12)))
                for other in range(7):
                    row[f'Cosine_To_C{other+1}_Before']=float(before_cos[c,other]);row[f'Cosine_To_C{other+1}_After']=float(after_cos[c,other]);row[f'Cosine_To_C{other+1}_Change']=float(after_cos[c,other]-before_cos[c,other])
                shift_rows.append(row)
        # Explicit C-B rows retain per-seed paired changes.
        b,c=seed_results['B'],seed_results['C']
        for i,j in PAIRS:
            for gt,pred in ((i,j),(j,i)):confusion_rows.append(dict(Seed=seed,Method='Delta_C_minus_B',GT_Class=gt,Pred_Class=pred,Hard_Count=int(c['hard'][gt-1,pred-1]-b['hard'][gt-1,pred-1]),Hard_Conditional_Probability=c['norm'][gt-1,pred-1]-b['norm'][gt-1,pred-1],Mean_Soft_Probability=c['soft'][gt-1,pred-1]-b['soft'][gt-1,pred-1]))
        for gt in range(7):
            for pred in range(7):mass_rows.append(dict(Seed=seed,Method='Delta_C_minus_B',GT_Class=gt+1,Probability_Class=pred+1,Mean_Probability=c['soft'][gt,pred]-b['soft'][gt,pred]))
        bg={(r['Class_i'],r['Class_j']):r for r in b['geometry']};cg={(r['Class_i'],r['Class_j']):r for r in c['geometry']}
        for key in bg:geometry_rows.append({'Seed':seed,'Method':'Delta_C_minus_B','Class_i':key[0],'Class_j':key[1],**{k:cg[key][k]-bg[key][k] for k in ('centroid_cosine','centroid_euclidean','radius_i','radius_j','inter_intra_ratio')}})
    write_csv(OUT/'class_pair_confusion.csv',confusion_rows);write_csv(OUT/'probability_mass_transfer.csv',mass_rows);write_csv(OUT/'centroid_geometry.csv',geometry_rows);write_csv(OUT/'classwise_shift_response.csv',shift_rows)
    summary=[]
    summarize_deltas(confusion_rows,['GT_Class','Pred_Class'],['Hard_Conditional_Probability','Mean_Soft_Probability'],summary,'Class-pair confusion')
    summarize_deltas(mass_rows,['GT_Class','Probability_Class'],['Mean_Probability'],summary,'Probability mass')
    summarize_deltas(geometry_rows,['Class_i','Class_j'],['centroid_cosine','centroid_euclidean','inter_intra_ratio'],summary,'Target feature geometry')
    # Shift response is shared transformation but learned feature response differs B/C.
    for method in METHODS:
        for cls in range(1,8):
            v=[r for r in shift_rows if r['Method']==method and r['Class']==cls]
            for metric in ('Centroid_Euclidean_Movement','Centroid_Cosine_Movement','Spread_Change','High_Alpha_Change_Fraction'):
                a=np.array([float(x[metric]) for x in v]);summary.append({'Analysis':'SceneShift response','Method':method,'Class':cls,'Metric':metric,'Mean':a.mean(),'Std':a.std(ddof=0),'Positive_Count':int((a>0).sum()),'Negative_Count':int((a<0).sum()),'Seeds':3})
    write_csv(OUT/'cross_seed_summary.csv',summary);(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('saved',OUT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=256);args=p.parse_args();main()
