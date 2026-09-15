import run_official_mluda_reproduction as b, pathlib, json, shutil, sys, ast
H=pathlib.Path(__file__).resolve().parent
for mode in ['post','pre']:
 base=H/'runs_mluda_official_reproduction_v1'; out=H/f'runs_shanghai_{mode}_supervised_seed1341'
 b.BASE=out
 b.SOURCE_REPO=H/'runs_mluda_official_reproduction_v1'
 # Stage the already-audited official snapshot into this independent root.
 snap_src=base/'shanghai_hangzhou'/'source_snapshot'
 snap_dst=out/'shanghai_hangzhou'/'source_snapshot'
 if not snap_dst.exists():
  snap_dst.mkdir(parents=True,exist_ok=True)
  for f in snap_src.glob('*.py'): shutil.copy2(f,snap_dst/f.name)
 def instr(src,mode=mode):
  src=src.replace('seeds = seeds[:3]\nnDataSet = len(seeds)','seeds = [1341]\nnDataSet = 1')
  if mode=='pre':
   marker='data_s,data_t = ILDA(data_s,data_t,pca_n,radius)'
   src=src.replace(marker,"""# PRE-ILDA SceneShift with raw-space target stats
sm=data_s.reshape(-1,data_s.shape[-1]).mean(0); ss=data_s.reshape(-1,data_s.shape[-1]).std(0); tm=data_t.reshape(-1,data_t.shape[-1]).mean(0); ts=data_t.reshape(-1,data_t.shape[-1]).std(0)
data_s=(data_s-sm)/(ss+1e-5)*(0.8*ts+0.2*ss)+0.8*tm+0.2*sm
_shift_loss=torch.tensor(0.0)
"""+marker)
  else:
   marker='data_s,data_t = ILDA(data_s,data_t,pca_n,radius)'
   src=src.replace(marker,marker+"\n_sm=data_s.reshape(-1,data_s.shape[-1]).mean(0); _ss=data_s.reshape(-1,data_s.shape[-1]).std(0); _tm=data_t.reshape(-1,data_t.shape[-1]).mean(0); _ts=data_t.reshape(-1,data_t.shape[-1]).std(0)")
   marker='            loss = cls_loss + 0.01 * lambd * lmmd_loss + contrastive_loss_t + contrastive_loss_s'
   add="""\n            _shifted_batch=(source_data.cuda()-torch.tensor(_sm,device='cuda')[None,:,None,None])/(torch.tensor(_ss,device='cuda')[None,:,None,None]+1e-5)*(0.8*torch.tensor(_ts,device='cuda')[None,:,None,None]+0.2*torch.tensor(_ss,device='cuda')[None,:,None,None])+0.8*torch.tensor(_tm,device='cuda')[None,:,None,None]+0.2*torch.tensor(_sm,device='cuda')[None,:,None,None]\n            _shift_out=feature_encoder(_shifted_batch,_shifted_batch)[3]\n            _shift_ce=crossEntropy(_shift_out,source_label.cuda())\n            loss = loss + 0.5 * _shift_ce"""
   add=add.replace('_shift_ce=crossEntropy(_shift_out,source_label.cuda())\n            loss = loss + 0.5 * _shift_ce','_shift_loss=0.5*crossEntropy(_shift_out,source_label.cuda())\n            loss = loss + _shift_loss')
   src=src.replace(marker,marker+add)
  return b.instrument(src,'config_SH2HZ')
 def conf(m,d,r,mode=mode):
  m.update(scene_shift_mode=mode,alpha=.8,lambda_shift=.5,preprocessing_note='independent quick counterpart runner; supervised shifted CE only' if mode=='post' else 'raw-space affine before official ILDA')
  (r/'config.json').write_text(json.dumps(m,indent=2))
 class E: pass
 e=E(); e.BASE=out; e.SOURCE_REPO=base; e.instrument=instr; e.configure=conf; e.hooks=lambda d,r:{}; e.SUMMARY=H/'summarize_official_mluda_reproduction.py'
 try: b.run('shanghai_hangzhou',False,extension=e)
 except Exception: raise
