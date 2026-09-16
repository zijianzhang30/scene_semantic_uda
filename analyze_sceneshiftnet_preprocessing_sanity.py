"""Compare loader normalization and raw cubes for own-model pure SceneShift."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import train_sceneshiftnet_dcrn_pavia as pavia
import train_sceneshiftnet_dcrn_shanghai as shanghai


HERE = Path(__file__).resolve().parent


def mean_distance(a, b):
    return float(np.linalg.norm(a.mean(0) - b.mean(0)))


def coral_distance(a, b):
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    return float(np.linalg.norm(ca - cb, "fro") / (4 * a.shape[1] ** 2))


def mmd_rbf(a, b):
    x, y = torch.from_numpy(a).float(), torch.from_numpy(b).float()
    combined = torch.cat((x, y))
    with torch.no_grad():
        distances = torch.cdist(combined, combined).square()
        median = torch.median(distances[distances > 0])
        n = len(x)
        values = []
        for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
            kernel = torch.exp(-distances / (2 * median * scale + 1e-12))
            values.append(
                kernel[:n, :n].mean() + kernel[n:, n:].mean()
                - 2 * kernel[:n, n:].mean()
            )
    return float(torch.stack(values).mean())


def sam(a, b):
    ma, mb = a.mean(0), b.mean(0)
    cosine = ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-12)
    return float(np.arccos(np.clip(cosine, -1, 1)))


def distances(a, b):
    return {
        "mean_distance": mean_distance(a, b),
        "coral_distance": coral_distance(a, b),
        "mmd_rbf": mmd_rbf(a, b),
        "sam_radians": sam(a, b),
    }


def analyze(name, loader, bands, samples, seed):
    rows = []
    for normalization in ("loader", "none"):
        source, _source_gt, target, _target_gt, _paths = loader(normalization)
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        source_flat = source.reshape(-1, bands)
        target_flat = target.reshape(-1, bands)
        sm, ss = source_flat.mean(0), source_flat.std(0)
        tm, ts = target_flat.mean(0), target_flat.std(0)
        shifted = (source_flat - sm) / (ss + 1e-5)
        shifted = shifted * (0.8 * ts + 0.2 * ss) + 0.8 * tm + 0.2 * sm

        delta = shifted - source_flat
        summary = {
            "dataset": name,
            "normalization": normalization,
            "use_ilda": False,
            "R_shift": float(np.linalg.norm(delta) / (np.linalg.norm(source_flat) + 1e-12)),
            "mean_abs_shift": float(np.mean(np.abs(delta))),
            "mean_delta_mu": float(np.mean(np.abs(shifted.mean(0) - sm))),
            "mean_delta_sigma": float(np.mean(np.abs(shifted.std(0) - ss))),
        }
        rng = np.random.RandomState(seed)
        source_ids = rng.choice(len(source_flat), min(samples, len(source_flat)), replace=False)
        target_ids = rng.choice(len(target_flat), min(samples, len(target_flat)), replace=False)
        sampled_source = source_flat[source_ids].astype(np.float32)
        sampled_shifted = shifted[source_ids].astype(np.float32)
        sampled_target = target_flat[target_ids].astype(np.float32)
        before, after = distances(sampled_source, sampled_target), distances(sampled_shifted, sampled_target)
        for metric in before:
            rows.append({
                **summary,
                "metric": metric,
                "d_source_target": before[metric],
                "d_shifted_target": after[metric],
                "relative_change_percent": 100 * (after[metric] - before[metric]) / (before[metric] + 1e-30),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1341)
    args = parser.parse_args()
    rows = analyze("pavia_u_pavia_c", pavia.load_cubes, 102, args.samples, args.seed)
    rows += analyze("shanghai_hangzhou", shanghai.load_cubes, 198, args.samples, args.seed)
    output = HERE / "runs_sceneshiftnet_dcrn/preprocessing_sanity_noilda"
    output.mkdir(parents=True, exist_ok=True)
    (output / "sanity.json").write_text(json.dumps(rows, indent=2))
    with (output / "sanity.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Dataset | Normalization | R_shift | mean|shift| | mean delta mu | mean delta sigma | Metric | d(S,T) | d(Sprime,T) | Relative Change")
    for row in rows:
        print(row["dataset"], row["normalization"], row["R_shift"], row["mean_abs_shift"], row["mean_delta_mu"], row["mean_delta_sigma"], row["metric"], row["d_source_target"], row["d_shifted_target"], row["relative_change_percent"])


if __name__ == "__main__":
    main()
