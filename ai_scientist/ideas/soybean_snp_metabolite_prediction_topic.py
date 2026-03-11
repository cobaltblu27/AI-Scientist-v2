"""
Starter code for the local soybean SNP -> metabolite task.

Guardrails for this task:
1. Main experiments must use the provided soybean files from the local workspace.
2. Do not replace the main evaluation with unrelated Hugging Face/public datasets.
3. Do not replace the main evaluation with synthetic data except for isolated debugging.
4. Keep genotype/label sample alignment explicit and logged.
"""

import gzip
import heapq
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

TARGET_COLUMNS = [
    "daidzein_values",
    "glycitein_values",
    "genistein_values",
    "glycitin",
    "malonyl-daidzin_values",
    "Malonyl-glycitin_values",
    "Malonyl-genistin_values",
    "daidzin_value",
    "genistin_value",
]

DEFAULT_MAX_SNPS = int(os.environ.get("SOYBEAN_MAX_SNPS", "4096"))
DEFAULT_MIN_MAF = float(os.environ.get("SOYBEAN_MIN_MAF", "0.01"))
DEFAULT_MAX_MISSING_RATE = float(os.environ.get("SOYBEAN_MAX_MISSING_RATE", "0.15"))


def normalize_sample_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace(" ", "_").replace("-", "_").replace("/", "_")
    name = re.sub(r"_+", "_", name)
    return name.lower()


def _iter_candidate_dataset_dirs():
    cwd = Path(os.getcwd()).resolve()
    env_dir = os.environ.get("SOYBEAN_DATA_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir).resolve())

    candidates.extend(
        [
            cwd,
            cwd / "input",
            cwd / "data",
            cwd.parent,
            cwd.parent / "input",
            cwd.parent / "data",
            cwd.parent.parent,
            cwd.parent.parent / "input",
            cwd.parent.parent / "data",
        ]
    )

    expanded = []
    for candidate in candidates:
        expanded.append(candidate)
        expanded.append(candidate / "np")

    seen = set()
    for candidate in expanded:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        yield candidate


def resolve_soybean_paths():
    for dataset_dir in _iter_candidate_dataset_dirs():
        snp_gz = dataset_dir / "SNP_data.tsv.gz"
        snp_tsv = dataset_dir / "SNP_data.tsv"
        label_path = dataset_dir / "soybean_label_map_dedup.csv"
        reference_path = dataset_dir / "Gmax_275_v2.0.softmasked_filtered.fa"

        if label_path.exists() and reference_path.exists() and (snp_gz.exists() or snp_tsv.exists()):
            return {
                "dataset_dir": dataset_dir,
                "snp_path": snp_gz if snp_gz.exists() else snp_tsv,
                "label_path": label_path,
                "reference_path": reference_path,
            }

    raise FileNotFoundError(
        "Could not find the soybean dataset files. Expected local files such as "
        "SNP_data.tsv.gz, soybean_label_map_dedup.csv, and "
        "Gmax_275_v2.0.softmasked_filtered.fa in ./input, ../input, ../../input, "
        "./data, or a directory pointed to by SOYBEAN_DATA_DIR."
    )


def resolve_shared_cache_dir() -> Path:
    cwd = Path(os.getcwd()).resolve()
    candidates = [
        cwd / "working",
        cwd.parent / "working",
        cwd.parent.parent / "working",
    ]
    for candidate in candidates:
        if candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    fallback = Path(working_dir)
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def peek_reference_headers(reference_path: Path, max_headers: int = 5):
    headers = []
    with open(reference_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                headers.append(line[1:].strip())
                if len(headers) >= max_headers:
                    break
    return headers


def encode_genotype_call(call: str) -> float:
    call = str(call).strip()
    if call in {"./.", ".|.", ".", ""}:
        return np.nan
    parts = re.split(r"[|/]", call)
    if not parts or any(part == "." for part in parts):
        return np.nan
    return float(sum(int(part) for part in parts))


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_nanmean(values) -> float:
    values = np.asarray(values, dtype=np.float32)
    if np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


def load_targets(label_path: Path):
    label_df = pd.read_csv(label_path)
    label_df["sample_key"] = label_df["library_name"].map(normalize_sample_name)
    if label_df["sample_key"].duplicated().any():
        dupes = label_df.loc[label_df["sample_key"].duplicated(), "library_name"].tolist()
        raise ValueError(f"Duplicate normalized sample names in labels: {dupes[:10]}")

    y = label_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    constant_mask = np.std(y, axis=0) < 1e-12
    return label_df, y, constant_mask


def stream_top_snps(
    snp_path: Path,
    sample_keys,
    cache_dir: Path,
    max_snps: int = DEFAULT_MAX_SNPS,
    min_maf: float = DEFAULT_MIN_MAF,
    max_missing_rate: float = DEFAULT_MAX_MISSING_RATE,
):
    cache_name = (
        f"soybean_top_snps_m{max_snps}_maf{int(min_maf * 1000)}"
        f"_miss{int(max_missing_rate * 1000)}.npz"
    )
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        print(f"Loaded cached SNP subset from {cache_path}")
        return cache["X"], cache["snp_ids"]

    opener = gzip.open if snp_path.suffix == ".gz" else open
    heap = []
    variants_seen = 0
    variants_kept = 0

    with opener(snp_path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        genotype_sample_names = header[4:]
        genotype_name_to_index = {
            normalize_sample_name(name): idx for idx, name in enumerate(genotype_sample_names)
        }
        missing = [key for key in sample_keys if key not in genotype_name_to_index]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} label samples in genotype matrix. "
                f"Examples: {missing[:10]}"
            )
        sample_indices = [genotype_name_to_index[key] for key in sample_keys]

        for line_idx, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            genotypes = fields[4:]
            dosage = np.empty(len(sample_indices), dtype=np.float32)

            for out_idx, sample_idx in enumerate(sample_indices):
                dosage[out_idx] = encode_genotype_call(genotypes[sample_idx])

            missing_mask = np.isnan(dosage)
            missing_rate = float(missing_mask.mean())
            if missing_rate > max_missing_rate:
                continue

            observed = dosage[~missing_mask]
            if observed.size == 0:
                continue

            observed_mean = float(observed.mean())
            dosage[missing_mask] = observed_mean

            allele_freq = observed_mean / 2.0
            maf = min(allele_freq, 1.0 - allele_freq)
            if maf < min_maf:
                continue

            variance = float(dosage.var())
            if variance <= 1e-8:
                continue

            variant_id = f"{fields[0]}:{fields[1]}:{fields[2]}>{fields[3]}"
            entry = (variance, variant_id, dosage.copy())

            if len(heap) < max_snps:
                heapq.heappush(heap, entry)
                variants_kept += 1
            elif variance > heap[0][0]:
                heapq.heapreplace(heap, entry)

            variants_seen += 1
            if line_idx % 50000 == 0:
                print(
                    f"Processed {line_idx:,} variants | "
                    f"eligible={variants_seen:,} | retained_top={min(len(heap), max_snps):,}"
                )

    selected = sorted(heap, key=lambda item: item[0], reverse=True)
    if not selected:
        raise RuntimeError("No SNPs survived filtering. Relax the MAF/missingness thresholds.")

    X = np.stack([item[2] for item in selected], axis=1).astype(np.float32)
    snp_ids = np.asarray([item[1] for item in selected], dtype=object)

    np.savez_compressed(cache_path, X=X, snp_ids=snp_ids)
    print(
        f"Saved cached SNP subset to {cache_path} | "
        f"eligible={variants_seen:,} | selected={X.shape[1]:,}"
    )
    return X, snp_ids


def run_ridge_cv(X: np.ndarray, y: np.ndarray, trait_names):
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_predictions = np.zeros_like(y, dtype=np.float32)
    variable_mask = np.std(y, axis=0) > 1e-12
    variable_indices = np.where(variable_mask)[0]

    if len(variable_indices) == 0:
        raise RuntimeError("All targets are constant; there is nothing to model.")

    fold_mean_pearson = []
    fold_mean_rmse = []
    fold_mean_r2 = []
    per_trait_metrics = {
        trait: {"pearson": [], "rmse": [], "r2": []} for trait in trait_names
    }

    constant_values = y.mean(axis=0)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), start=1):
        model = Pipeline(
            [
                ("x_scaler", StandardScaler()),
                ("ridge", Ridge(alpha=10.0, random_state=RANDOM_SEED)),
            ]
        )

        y_train_log = np.log1p(y[train_idx][:, variable_indices])
        model.fit(X[train_idx], y_train_log)

        pred_log = model.predict(X[val_idx])
        pred_var = np.expm1(pred_log)
        pred_var = np.clip(pred_var, a_min=0.0, a_max=None).astype(np.float32)

        pred_full = np.tile(constant_values, (len(val_idx), 1)).astype(np.float32)
        pred_full[:, variable_indices] = pred_var
        cv_predictions[val_idx] = pred_full

        pearsons = []
        rmses = []
        r2s = []
        for trait_idx, trait_name in enumerate(trait_names):
            y_true = y[val_idx, trait_idx]
            y_pred = pred_full[:, trait_idx]
            pearson = safe_pearsonr(y_true, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2 = float("nan")
            if np.std(y_true) > 1e-12:
                r2 = float(r2_score(y_true, y_pred))

            per_trait_metrics[trait_name]["pearson"].append(pearson)
            per_trait_metrics[trait_name]["rmse"].append(rmse)
            per_trait_metrics[trait_name]["r2"].append(r2)

            if not np.isnan(pearson):
                pearsons.append(pearson)
            rmses.append(rmse)
            if not np.isnan(r2):
                r2s.append(r2)

        fold_mean_pearson.append(float(np.mean(pearsons)))
        fold_mean_rmse.append(float(np.mean(rmses)))
        fold_mean_r2.append(float(np.mean(r2s)))

        print(
            f"Fold {fold_idx}: "
            f"mean_pearson={fold_mean_pearson[-1]:.4f}, "
            f"mean_rmse={fold_mean_rmse[-1]:.4f}, "
            f"mean_r2={fold_mean_r2[-1]:.4f}"
        )

    final_model = Pipeline(
        [
            ("x_scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0, random_state=RANDOM_SEED)),
        ]
    )
    final_model.fit(X, np.log1p(y[:, variable_indices]))
    coefficients = final_model.named_steps["ridge"].coef_

    return {
        "cv_predictions": cv_predictions,
        "variable_mask": variable_mask,
        "fold_mean_pearson": fold_mean_pearson,
        "fold_mean_rmse": fold_mean_rmse,
        "fold_mean_r2": fold_mean_r2,
        "per_trait_metrics": per_trait_metrics,
        "final_coefficients": coefficients,
        "variable_indices": variable_indices,
    }


def build_top_snp_summary(trait_names, variable_indices, coefficients, snp_ids, top_k: int = 20):
    top_snps = {}
    for coef_row, trait_idx in enumerate(variable_indices):
        order = np.argsort(np.abs(coefficients[coef_row]))[::-1][:top_k]
        top_snps[trait_names[trait_idx]] = [str(snp_ids[i]) for i in order]
    return top_snps


def plot_results(working_dir_path, y, results, trait_names):
    pearson_path = os.path.join(working_dir_path, "soybean_cv_metrics.png")
    scatter_path = os.path.join(working_dir_path, "soybean_prediction_scatter.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(results["fold_mean_pearson"], marker="o")
    axes[0].set_title("Fold Mean Pearson")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Pearson")

    axes[1].plot(results["fold_mean_rmse"], marker="o", color="tab:orange")
    axes[1].set_title("Fold Mean RMSE")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("RMSE")

    axes[2].plot(results["fold_mean_r2"], marker="o", color="tab:green")
    axes[2].set_title("Fold Mean R2")
    axes[2].set_xlabel("Fold")
    axes[2].set_ylabel("R2")
    plt.tight_layout()
    plt.savefig(pearson_path, dpi=160)
    plt.close(fig)

    variable_indices = np.where(results["variable_mask"])[0][:4]
    if len(variable_indices) > 0:
        fig, axes = plt.subplots(1, len(variable_indices), figsize=(4 * len(variable_indices), 4))
        if len(variable_indices) == 1:
            axes = [axes]
        for ax, trait_idx in zip(axes, variable_indices):
            ax.scatter(
                y[:, trait_idx],
                results["cv_predictions"][:, trait_idx],
                alpha=0.7,
                edgecolor="none",
            )
            lim_min = min(y[:, trait_idx].min(), results["cv_predictions"][:, trait_idx].min())
            lim_max = max(y[:, trait_idx].max(), results["cv_predictions"][:, trait_idx].max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="black")
            ax.set_title(trait_names[trait_idx])
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=160)
        plt.close(fig)

    return [pearson_path, scatter_path]


print("Starting soybean real-world starter baseline.")
print("This code is intended to keep the main experiment anchored to the local soybean dataset.")

paths = resolve_soybean_paths()
print(
    json.dumps(
        {
            "dataset_dir": str(paths["dataset_dir"]),
            "snp_path": str(paths["snp_path"]),
            "label_path": str(paths["label_path"]),
            "reference_path": str(paths["reference_path"]),
        },
        indent=2,
    )
)

shared_cache_dir = resolve_shared_cache_dir()
reference_headers = peek_reference_headers(paths["reference_path"])
print(f"Reference headers preview: {reference_headers}")

label_df, y, constant_mask = load_targets(paths["label_path"])
sample_keys = label_df["sample_key"].tolist()
print(f"Loaded labels: {label_df.shape[0]} samples, {y.shape[1]} targets")
print(
    "Constant targets detected: "
    + ", ".join(
        trait for trait, is_constant in zip(TARGET_COLUMNS, constant_mask) if is_constant
    )
)

X, snp_ids = stream_top_snps(
    snp_path=paths["snp_path"],
    sample_keys=sample_keys,
    cache_dir=shared_cache_dir,
    max_snps=DEFAULT_MAX_SNPS,
    min_maf=DEFAULT_MIN_MAF,
    max_missing_rate=DEFAULT_MAX_MISSING_RATE,
)
print(f"Selected SNP matrix shape: {X.shape}")

results = run_ridge_cv(X, y, TARGET_COLUMNS)
top_snps = build_top_snp_summary(
    TARGET_COLUMNS,
    results["variable_indices"],
    results["final_coefficients"],
    snp_ids,
)
plot_paths = plot_results(working_dir, y, results, TARGET_COLUMNS)

experiment_data = {
    "soybean_realworld": {
        "metrics": {
            "fold_mean_pearson": results["fold_mean_pearson"],
            "fold_mean_rmse": results["fold_mean_rmse"],
            "fold_mean_r2": results["fold_mean_r2"],
            "per_trait_metrics": results["per_trait_metrics"],
        },
        "predictions": results["cv_predictions"],
        "ground_truth": y,
        "sample_names": label_df["library_name"].tolist(),
        "trait_names": TARGET_COLUMNS,
        "selected_snp_ids": snp_ids.tolist(),
        "variable_target_mask": results["variable_mask"].tolist(),
        "top_snps_by_trait": top_snps,
        "plots": plot_paths,
    }
}

summary = {
    "dataset_dir": str(paths["dataset_dir"]),
    "num_samples": int(X.shape[0]),
    "num_selected_snps": int(X.shape[1]),
    "targets": TARGET_COLUMNS,
    "constant_targets": [
        trait for trait, is_constant in zip(TARGET_COLUMNS, constant_mask) if is_constant
    ],
    "reference_headers_preview": reference_headers,
    "top_snps_by_trait": top_snps,
}

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "soybean_realworld_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print("\nPer-trait cross-validated metrics:")
for trait_name in TARGET_COLUMNS:
    metrics = results["per_trait_metrics"][trait_name]
    print(
        f"{trait_name}: "
        f"pearson={safe_nanmean(metrics['pearson']):.4f}, "
        f"rmse={safe_nanmean(metrics['rmse']):.4f}, "
        f"r2={safe_nanmean(metrics['r2']):.4f}"
    )

print("\nArtifacts written to:")
print(f"- {os.path.join(working_dir, 'experiment_data.npy')}")
print(f"- {os.path.join(working_dir, 'soybean_realworld_summary.json')}")
for plot_path in plot_paths:
    print(f"- {plot_path}")
