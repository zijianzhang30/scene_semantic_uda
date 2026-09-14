"""Summarize only the isolated official baseline run, including partial status."""
import csv
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs_mluda_official_reproduction_v1'
EXPECTED = {'pavia': [1622,1322,1256], 'shanghai_hangzhou': [1341,1535,1631]}
rows=[]; summary={}
for dataset,seeds in EXPECTED.items():
    selected=[]
    for seed in seeds:
        path=ROOT/dataset/('seed_'+str(seed))/'result.json'
        if path.exists():
            row=json.loads(path.read_text())
            assert row['seed']==seed and row['epoch']==100 and row['units']=='percent'
            selected.append(row); rows.append(row)
    summary[dataset]={'completed':len(selected),'expected':len(seeds),'seeds':seeds,
                      'paper_reported':None,'gap_to_paper':None}
    if selected:
        summary[dataset]['metrics']={k:{'mean':statistics.mean(r[k] for r in selected),
                                       'std':statistics.pstdev(r[k] for r in selected)}
                                     for k in ('oa','aa','kappa')}
        summary[dataset]['per_class']=[{'mean':statistics.mean(r['per_class_accuracy'][i] for r in selected),
                                      'std':statistics.pstdev(r['per_class_accuracy'][i] for r in selected)}
                                     for i in range(len(selected[0]['per_class_accuracy']))]
ROOT.mkdir(exist_ok=True)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
with (ROOT/'per_seed.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['dataset','seed','OA','AA','Kappa','epoch'])
    w.writerows([r['dataset'],r['seed'],r['oa'],r['aa'],r['kappa'],r['epoch']] for r in rows)
with (ROOT/'per_class.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['dataset','seed','class','accuracy_percent'])
    for r in rows:
        w.writerows([r['dataset'],r['seed'],i+1,v] for i,v in enumerate(r['per_class_accuracy']))
lines=['# MLUDA 官方协议复现：Pavia / Shanghai', '',
       '仅 Original MLUDA。直接执行官方入口的隔离快照，增加日志/结果保存，去除最后绘图；未启动 SceneShift。', '',
       '使用每个数据集前3个官方 seeds；同进程先执行一次 ILDA，再进入官方 seed 循环。',
       '使用 GT>0 target pool 和官方 class-grouped shuffle；无 target class loss supervision，但访问 target mask 和用于排序的类别。',
       'SGD 每epoch重建，100 epochs；评估 epoch100，保留官方 drop_last、分母及最后 source batch reference。',
       'Pavia: lr=.001, patch11, 102 bands; Shanghai: lr=.0003, patch1, 198 bands。', '',
       'std 使用 ddof=0。partial 行不可当作3-seed最终结果。', '',
       '| Dataset | Completed | OA | AA | Kappa |', '|---|---:|---:|---:|---:|']
for d,s in summary.items():
    vals=[f"{s['metrics'][k]['mean']:.2f}±{s['metrics'][k]['std']:.2f}" for k in ('oa','aa','kappa')] if s['completed'] else ['Pending']*3
    lines.append('| '+d+' | '+str(s['completed'])+'/3 | '+' | '.join(vals)+' |')
lines+=['','## 每 seed 原始结果','','| Dataset | Seed | OA | AA | Kappa | Epoch |','|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['dataset']} | {r['seed']} | {r['oa']:.4f} | {r['aa']:.4f} | {r['kappa']:.4f} | 100 |")
lines+=['','## Per-class mean ± std','']
for d,s in summary.items():
    if s['completed']:
        lines.append(d+': '+', '.join(f"C{i+1} {v['mean']:.2f}±{v['std']:.2f}" for i,v in enumerate(s['per_class'])))
lines+=['','## 论文参照与 SceneShift 阶段门槛','',
        'TODO：核实 DOI 10.1109/TGRS.2024.3407952 的原文表号、精确指标、数据版本和代码发布版本；不以用户口述“90%+”替代已核实数字。',
        '尚未形成可靠 paper gap，不能认定已经接近官方结果。SceneShift 阶段保持未启动。',
        '各dataset目录保存 config.json、source_snapshot、executed_entry.py、数据/源码SHA256和git_commit.txt。',
        'local repo HEAD 包含本地开发提交；作者原始发布commit等价性未独立验证。',
        '不与 clean-protocol 历史结果混合统计。']
(HERE/'mluda_official_reproduction_pavia_shanghai.md').write_text('\n'.join(lines)+'\n')
print('Updated official reproduction report',flush=True)
