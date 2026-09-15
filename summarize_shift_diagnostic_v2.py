"""Summarize existing diagnostics only; never runs a model."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE/'runs_official_shift_diagnostic'
def avg(bs,key): return float(np.mean([b[key] for b in bs]))
def table(header,rows):
    return ['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(f'{v:.5f}' if isinstance(v,float) else str(v) for v in r)+' |' for r in rows]
def main():
    ds={1622:json.loads((ROOT/'pavia_seed1622_v2/result.json').read_text()),1256:json.loads((ROOT/'pavia_seed1256/result.json').read_text())}
    common=['本轮不训练、不做optimizer step、不更新模型权重；临时BN变化只在内存中，诊断结束state_dict及原checkpoint SHA256均验证不变。',
      '每seed四个固定32样本batch，128次source训练样本抽样，跨batch可能重复。不是验证集或全source结论。梯度在train模式下计算，eval前重载全部buffer。所有gradient cosine是final checkpoint局部值，不能倒推整段训练因果。',
      'BN delta指原path之后，额外path的三次paired forward引起的累计变化；各共享BN通常更新6次，不是optimizer单步或单个BN调用。']
    rows=[]; layer=[]; cls=[]
    for seed,d in ds.items():
        for model,m in d['models'].items():
            bs=m['batches']; cos=[b['gradient_cosine'] for b in bs]
            bm=np.mean([b['extra_branch_bn_delta']['aggregate']['running_mean']['mean_abs'] for b in bs]);bv=np.mean([b['extra_branch_bn_delta']['aggregate']['running_var']['mean_abs'] for b in bs])
            rows.append([seed,model,avg(bs,'transport_only_abs_delta'),avg(bs,'total_shift_abs_delta'),avg(bs,'original_grad_norm'),avg(bs,'weighted_extra_grad_norm'),avg(bs,'weighted_extra_grad_norm_ratio'),np.mean(cos),f'{min(cos):.4f}…{max(cos):.4f}',sum(c<0 for c in cos),float(bm),float(bv)])
            for group in ['backbone','classifier','projection_heads']:
                vals=[b['layer_gradients'][group] for b in bs]; valid=[v['cosine'] for v in vals if v['cosine'] is not None]
                layer.append([seed,model,group,float(np.mean([v['original_norm'] for v in vals])),float(np.mean([v['weighted_extra_norm'] for v in vals])),float(np.mean([v['ratio'] for v in vals])),float(np.mean(valid)) if valid else 'undefined (zero gradient)'])
            for k in range(1,8):
                vals=[b['per_class'][str(k)] for b in bs]
                cls.append([seed,model,k,float(np.mean([v['affine_abs_delta'] for v in vals])),float(np.mean([v['shift_abs_delta'] for v in vals])),float(np.mean([v['raw_source_accuracy'] for v in vals])),float(np.mean([v['shift_pseudo_source_agreement'] for v in vals]))])
    a=['# Seed1622 vs seed1256 frozen diagnostic','']+common+['']
    a+=table(['seed','model','affine Δ','noisy Δ','original grad norm','gamma-extra grad norm','ratio','cos mean','cos range','negative/4','BN mean absΔ','BN var absΔ'],rows)
    a+=['','## 分层梯度','']+table(['seed','model','group','original norm','weighted extra norm','ratio','cos'],layer)
    a+=['','## Per-class shift / pseudo consistency','']+table(['seed','model','class','affine Δ','noisy Δ','raw acc','pseudo agreement'],cls)
    a+=['','## Reproduction','']
    for seed,d in ds.items():
        a.append(f"seed{seed} cached stats max differences: {d['target_support']['saved_vs_reconstructed_stats_max']}")
        for model,m in d['models'].items():
            e=m['reference_evaluation']['official_last']; a.append(f"{model}: OA {e['oa_official_denominator']:.5f}, AA {e['aa']:.5f}, Kappa {e['kappa']:.5f}; prediction agreement {e['original_saved_prediction_agreement']:.5f}; state/checkpoint unchanged.")
    seed_conclusion='''
## Seed结论

Original→Shift正式固定epoch负迁移：1622 ΔOA/AA/Kappa = −2.4724/−2.9126/−2.9573；1256 = −7.3078/−5.3415/−8.6285。

1256复现了“affine变化弱、Full-shift语义保持、额外梯度不小”的特征，但不是完全一样：Shift checkpoint的norm ratio从1622的0.5518上升至0.8653，negative cosine batch从0/4变成2/4（最小−0.237）。Original checkpoint也出现2/4负cosine。seed间source抽样、target batch及模型不同，不是单变量因果试验。

1256的projection-head平均ratio为1.4013（各batch ratio的均值，不等于均值norm相除），backbone为0.8031。classifier extra梯度为零：原LMMD的Weight.cal_weight通过.cpu().data.numpy()构建权重，pseudo被detach，额外SCL作用于projection/features；这符合现有官方实现，并非新增漏优化。

BN临时变化存在，但1256的mean/var变化并不大于1622，因此“最差seed是因为更大的单步BN扰动”未获支持。没有测试BN对最终OA的因果作用。每个训练态batch结束的buffer变化未持久化，评估前恢复原state_dict。

所有final predictions100%复现，cached stats逐元素复现；在所测source抽样上raw与Full-shift pseudo一致率均100%。这削弱ILDA漂移、评估reference、final pseudo collapse解释，不能排除训练早期或未测样本的问题。
'''
    a+=['',seed_conclusion]
    (ROOT/'pavia_seed1256/summary.md').write_text('\n'.join(a)+'\n')
    (ROOT/'pavia_seed1622_v2/summary.md').write_text('\n'.join(a)+'\n')
    o=json.loads((ROOT/'pavia_support_oracle/result.json').read_text())
    lines=['# Pavia target-support ORACLE diagnostic','',
      '**Mask与ClassBalanced使用target注释，只作事后机制诊断，不能作为正式无监督方法或target OA成绩。没有任何训练或参数更新。**',
      '两种seed/两种checkpoint；同source batch，同scale/noise RNG；alpha=.8，source stats不变。分别报告pure affine和既定噪声版。',
      'ClassBalanced用各类等权混合分布的总方差（含类间均值方差），不是平均std。谱距离使用patch中心像素；input MAE使用全部patch像素。',
      'Feature KNN：同层counterpart pre-classifier feature，L2 normalize，top10 mean cosine distance，越低越近；每个target support固定1024个patch。官方mask与full-bank不混合。小差值没有统计显著性保证。',
      '各类谱距离是源类别与官方shared-class target GT对应类别的oracle post-hoc比较；不进入训练。CORAL/MMD未新增实现，记未计算。','',
      '## Target support differences','',json.dumps(o['target_support_comparison'],indent=2),'', '## Transport/global center-spectral profile distances','']
    ts=[];fs=[];pc=[];pb=[]
    for seed,run in o['runs'].items():
        for version,t in run['transport'].items():
            ts.append([seed,version,t['input_mae'],t['spectral_proximity']['official_mask']['mean_profile_mae'],t['spectral_proximity']['official_mask']['std_profile_mae'],t['spectral_proximity']['full_cube']['mean_profile_mae'],t['spectral_proximity']['full_cube']['std_profile_mae']])
            for k,p in t['per_class'].items():
                pc.append([seed,version,k,p['input_mae'],p['target_class_profile_distance']['mean_profile_mae'],p['target_class_profile_distance']['std_profile_mae']])
        for model,m in run['models'].items():
            for version,v in m['versions'].items():
                fs.append([seed,model,version,v['feature_knn']['official_mask'],v['feature_knn']['full_cube'],v['pseudo_consistency'],v['raw_source_accuracy']])
                for k,p in v['per_class'].items():pb.append([seed,model,version,k,p['consistency'],p['feature_knn']['official_mask'],p['feature_knn']['full_cube']])
    lines+=table(['seed','version','input MAE','mask mean MAE','mask std MAE','full mean MAE','full std MAE'],ts)
    lines+=['','## Feature proximity + semantics','']+table(['seed','model','version','mask KNN','full KNN','pseudo consistency','raw source acc'],fs)
    lines+=['','## Per-class profile distance','']+table(['seed','version','class','input MAE','target class mean MAE','target class std MAE'],pc)
    lines+=['','## Per-class semantics + feature KNN','']+table(['seed','model','version','class','consistency','mask KNN','full KNN'],pb)
    oracle_conclusion='''
## Oracle机制结论（不作为方法成绩）

1. **Full与official mask支持分布确有偏差。** 全图/Mask平均band mean差0.22154、std差0.09052，远大于之前全图source/target残差；但这不单独证明它造成OA下降。
2. **Mask能让整体更target-like，但不能同时无损保留source语义。** 四个seed/checkpoint组合的pure-affine mask KNN都降低；语义一致率却从Full的100%降至89.84%–96.88%。固定噪声版趋势相同，不是随机噪声不一致造成。
3. **C2受损、C3受益的oracle trade-off明确。** C2 pure-affine Mask一致率分别为1622 Original45%/Shift35%，1256 Original80%/Shift45%；ClassBalanced提升到65%/55%/85%/60%，但远未回到Full100%。这些只是每seed20次C2 source抽样的分类一致率，不是target C2 accuracy。
4. **类条件谱方向不统一。** seed1256 C2 target-class均值profile MAE：Source .23432→Full .23747→Mask .29918→Balanced .31482，std也逐步变差；C3则 .68120→.68132→.51260→.43961，Mask/Balanced使其靠近对应target类且pseudo一致率保持100%。1622同趋势。C4均值也改善；C7均值改善但std/KNN并非一致改善。说明全局对齐不等于每类语义对齐。
5. **Full下“global closer but class-conditionally worse”不构成稳定证据。** 两seed Full-affine的全局谱mean MAE略变差，KNN大多略变差；只有1256 Shift模型mask KNN有极小改善(.03605→.03601)，不足以确认稳定global closer。Full的C2均值距离变差，C3均值微变差但std略改善。更清楚的global-closer/C2-worse现象出现在oracle Mask/Balanced，不应错归到当前Full方案。
6. **Balanced部分缓解而非消除类别偏置。** 四组合整体语义一致率都优于Mask（92.97%–97.66%），但feature KNN不是全部优于Mask，且C2谱profile更远；不能称其统一最优。更高consistency与更差C2谱距离也说明这些代理量并不等价。

这里距离使用固定source样本与固定1024-target-feature banks，无K/alpha/gamma sweep；没有置信区间或显著性检验。Mask/class标签是oracle事后信息，不能把这些数值包装成新的无监督结果。
'''
    lines+=['',oracle_conclusion]
    (ROOT/'pavia_support_oracle/summary.md').write_text('\n'.join(lines)+'\n')
    ranking='''
## 原因排序与下一步

排序区分“机制事实的证据强度”与“造成训练负迁移的因果证据”。本轮均为冻结final checkpoint，不能给出已证实训练因果。

1. **B transport弱但auxiliary优化负担强：较强支持。** 两seed affine约.0036、noisy约.026，而加权extra梯度为original的.55–.87；1256局部冲突更多。不是建议调gamma，也不等于所有extra梯度有害。
2. **C target-support mismatch：事实明确、因果部分支持。** Mask确实改善整体proximity，说明Full stats未代表official support；但oracle Mask损伤C2，直接替换不是已验证修复。
3. **D optimization side effect：局部梯度冲突有新支持；BN因果仍弱。** 1256两模型各2/4负cosine；BN单步幅度并不更大。不能将两种机制混称为已证实BN问题。
4. **A transport方向错误：仅类条件/方案相关支持，不是全局错误。** Full几乎identity，不宜说大幅反向transport；Mask/Balanced对C3更合理却对C2更差，单一全局矩映射有trade-off。
5. **E 多因素共同作用是目前最合理综合描述**，但非新增独立证据。不能只归因于mask、alpha或ILDA。

被削弱的假设：数据/ILDA重建错误（stats及预测复现）；所测reference导致下降（不变）；final Full pseudo collapse（未见）；所有seed梯度都无冲突（1256推翻）；更大的单步BN扰动解释最差seed（不支持）。训练早期伪标签/BN累计效应、source-val vs final checkpoint仍未排除。

下一步最小必要建议（本轮不执行）：

- **冻结权重BN-only诊断**：对checkpoint副本分开重放固定original/extra batch并仅保留临时BN buffers，比较同一target固定评价；不选最优buffer替代正式模型、不保存覆盖checkpoint。只改变buffer，隔离BN是否足以改变C2/整体预测。
- **固定batch分项梯度诊断**：把extra LMMD/source-SCL/shift-SCL单独测norm与原目标cosine，至少覆盖更多固定batch及C2/C3；如已有早期checkpoint才复用检查早期，缺失标TODO，不为诊断重训。先确认冲突来自何项，而不是继续改stats或调alpha/gamma。

交付：pavia_seed1256/result.json及summary.md；pavia_support_oracle/result.json、summary.md、per_band_stats.csv、per_class_stats.csv；1622新增BN/梯度记录独立保存pavia_seed1622_v2，未覆盖v1。
'''
    (HERE/'sceneshift_negative_transfer_diagnostic_v2.md').write_text('\n'.join(a)+'\n\n'+ '\n'.join(lines[:9])+ '\n\n## Full / Mask / ClassBalanced 概览\n\n'+'\n'.join(table(['seed','model','version','mask KNN','full KNN','pseudo consistency','raw source acc'],fs))+'\n\n'+oracle_conclusion+'\n'+ranking+'\n\n详细谱距离、C1–C7、per-band数据见 runs_official_shift_diagnostic/pavia_support_oracle/summary.md 和两个CSV。\n')
if __name__=='__main__':main()
