"""Post-ILDA SceneShift domain-gap analysis for Shanghai -> Hangzhou.

This is a read-only diagnostic of the completed official-style SceneShift v1
seed-1341 checkpoint. Target labels are loaded by the repository data helper
because they share DataCube.mat with the images, but are discarded immediately
and never used for sampling, distances, features, or plots.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE


HERE = Path(__file__).resolve().parent
MLUDA_REPO = Path("/home/zhangzj26/TGRS_MLUDA-2024")
RUN = HERE / "runs_mluda_official_sceneshift_v1" / "shanghai_hangzhou"
SNAPSHOT = RUN / "source_snapshot"
sys.path.insert(0, str(SNAPSHOT))

import utils  # noqa: E402
from UtilsCMS import ILDA  # noqa: E402
from net2 import DSANSS  # noqa: E402

# UtilsCMS requests SimHei globally, which is unavailable on this server.
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def mean_distance(a, b):
    return float(np.linalg.norm(a.mean(0) - b.mean(0)))


def coral_distance(a, b):
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    return float(np.linalg.norm(ca - cb, "fro") / (4 * a.shape[1] ** 2))


def mmd_rbf(a, b):
    x, y = torch.from_numpy(a).float(), torch.from_numpy(b).float()
    z = torch.cat((x, y))
    with torch.no_grad():
        squared_distance = torch.cdist(z, z).square()
        median = torch.median(squared_distance[squared_distance > 0])
        values = []
        n = len(x)
        for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
            kernel = torch.exp(-squared_distance / (2 * median * scale + 1e-12))
            values.append(
                kernel[:n, :n].mean()
                + kernel[n:, n:].mean()
                - 2 * kernel[:n, n:].mean()
            )
    return float(torch.stack(values).mean())


def mean_sam(a, b):
    ma, mb = a.mean(0), b.mean(0)
    cosine = ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-12)
    return float(np.arccos(np.clip(cosine, -1, 1)))


def metrics(a, b):
    return {
        "mean_distance": mean_distance(a, b),
        "coral_distance": coral_distance(a, b),
        "mmd_rbf": mmd_rbf(a, b),
        "mean_sam_radians": mean_sam(a, b),
    }


def plot_tsne(groups, path, title):
    values = np.concatenate(groups)
    domains = np.repeat(np.arange(3), [len(group) for group in groups])
    embedding = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=1341,
    ).fit_transform(values)
    for index, name in enumerate(("Source", "Shifted Source", "Target")):
        selected = domains == index
        plt.scatter(
            embedding[selected, 0], embedding[selected, 1], s=5, alpha=0.5, label=name
        )
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--tsne-samples", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    np_rng = np.random.RandomState(1341)
    torch.manual_seed(1341)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1341)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = RUN / "seed_1341" / "domain_gap_analysis_post_ilda"
    output.mkdir(parents=True, exist_ok=True)

    source, target, _source_gt, _target_gt = utils.cubeData(
        str(MLUDA_REPO / "datasets" / "Shanghai-Hangzhou" / "DataCube.mat")
    )
    # Match the analyzed training run exactly: SceneShift statistics and the
    # shift itself are computed after official ILDA.
    source, target = ILDA(source, target, 2, 0.00009)
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    bands = source.shape[-1]
    assert bands == 198 and target.shape[-1] == bands

    source_all = source.reshape(-1, bands)
    target_all = target.reshape(-1, bands)
    source_ids = np_rng.choice(len(source_all), args.samples, replace=False)
    target_ids = np_rng.choice(len(target_all), args.samples, replace=False)
    source_input = source_all[source_ids].astype(np.float32)
    target_input = target_all[target_ids].astype(np.float32)

    source_mean, source_std = source_all.mean(0), source_all.std(0)
    target_mean, target_std = target_all.mean(0), target_all.std(0)
    with torch.no_grad():
        x = torch.from_numpy(source_input).to(device)[:, :, None, None]
        to_tensor = lambda value: torch.as_tensor(
            value, device=device, dtype=x.dtype
        )[None, :, None, None]
        shifted = (x - to_tensor(source_mean)) / (to_tensor(source_std) + 1e-5)
        shifted = shifted * (
            0.8 * to_tensor(target_std) + 0.2 * to_tensor(source_std)
        ) + 0.8 * to_tensor(target_mean) + 0.2 * to_tensor(source_mean)
        # Exact saved Shanghai SceneShift v1 perturbations.
        shifted = shifted * (
            1 + 0.04 * torch.randn(x.size(0), 1, 1, 1, device=device)
        ) + 0.015 * F.avg_pool2d(torch.randn_like(shifted), 5, 1, 2)
        shifted_input = shifted[:, :, 0, 0].cpu().numpy()

    checkpoint_path = RUN / "seed_1341" / "final_epoch100.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DSANSS(198, 1, 3).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    def backbone_features(vectors):
        results = []
        captured = []

        def capture_attention_input(_module, inputs):
            # inputs[0] is the 288-D pooled source-side DCRN representation,
            # immediately before MLUDA CrossAttention.
            captured.append(inputs[0].detach())

        handle = model.feature_layers.atten.register_forward_pre_hook(
            capture_attention_input
        )
        try:
            with torch.no_grad():
                for start in range(0, len(vectors), args.batch_size):
                    batch = torch.from_numpy(
                        vectors[start : start + args.batch_size]
                    ).to(device)[:, :, None, None]
                    captured.clear()
                    model.feature_layers(batch, batch)
                    assert len(captured) == 1
                    results.append(captured[0].cpu().numpy())
        finally:
            handle.remove()
        result = np.concatenate(results)
        assert result.shape == (len(vectors), 288)
        return result.astype(np.float32)

    source_feature = backbone_features(source_input)
    shifted_feature = backbone_features(shifted_input)
    target_feature = backbone_features(target_input)

    rows = []
    for space, original, shifted_values, target_values in (
        ("input_spectral_post_ilda", source_input, shifted_input, target_input),
        ("dcrn_feature_pre_cross_attention", source_feature, shifted_feature, target_feature),
    ):
        before, after = metrics(original, target_values), metrics(
            shifted_values, target_values
        )
        for metric in before:
            rows.append(
                {
                    "space": space,
                    "metric": metric,
                    "d_source_target": before[metric],
                    "d_shifted_target": after[metric],
                    "relative_change_percent": 100
                    * (after[metric] - before[metric])
                    / before[metric],
                }
            )

    metadata = {
        "dataset": "Shanghai->Hangzhou",
        "seed": 1341,
        "checkpoint": str(checkpoint_path),
        "scene_shift_position": "post-ILDA",
        "alpha": 0.8,
        "random_scale": True,
        "smooth_noise": True,
        "statistics": "all pixels, per band, no GT mask",
        "input_dim": bands,
        "feature_dim": 288,
        "feature_layer": "DCRN pooled representation immediately before MLUDA CrossAttention",
        "samples_per_domain": args.samples,
        "target_gt_used": False,
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "domain_gap_metrics.json").write_text(json.dumps(rows, indent=2))
    with (output / "domain_gap_metrics.csv").open("w") as file:
        file.write("Space,Metric,d(S,T),d(Sprime,T),Relative Change (%)\n")
        for row in rows:
            file.write(
                f"{row['space']},{row['metric']},{row['d_source_target']},"
                f"{row['d_shifted_target']},{row['relative_change_percent']}\n"
            )

    count = min(args.tsne_samples, args.samples)
    plot_tsne(
        (source_input[:count], shifted_input[:count], target_input[:count]),
        output / "tsne_input_post_ilda.png",
        "Shanghai->Hangzhou input spectral space (post-ILDA SceneShift)",
    )
    plot_tsne(
        (source_feature[:count], shifted_feature[:count], target_feature[:count]),
        output / "tsne_feature.png",
        "Shanghai->Hangzhou DCRN feature space",
    )

    print("Space | Metric | d(S,T) | d(Sprime,T) | Relative Change")
    for row in rows:
        print(
            row["space"], row["metric"], row["d_source_target"],
            row["d_shifted_target"], row["relative_change_percent"]
        )


if __name__ == "__main__":
    main()
