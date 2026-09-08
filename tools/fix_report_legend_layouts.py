from pathlib import Path
import json
import os
import re
import shutil
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-3brain")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brain_3d.Connectivity import ChannelToGrid, ChannelValuesToGrid, SelectStrongEdges


MARK_ROOT = Path("/Users/cristina/Documents/GitHub/3D-BrAIn_codes/outputs/analisi_dati_Mark")
THREEBRAIN_ROOT = Path("/Users/cristina/Documents/GitHub/3D-BrAIn_codes/outputs/analisi_dati_3BRAIN")


def div_from_name(name):
    match = re.search(r"\bD(?:IV)?\s*[_-]?(\d+)\b", str(name), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def chip_from_name(name):
    match = re.search(r"chip\s*[_-]?(\d+)|Chip\s*[_-]?(\d+)", str(name), flags=re.IGNORECASE)
    if match:
        return match.group(1) or match.group(2)
    return "unknown"


def condition_3brain(name):
    text = str(name).lower()
    if "100um4ap_50umap5" in text or "50umap5_100um4ap" in text:
        return "excitation-inhibition"
    if "100um4ap" in text:
        return "excitation"
    if "50umap5" in text:
        return "inhibition"
    return "spontaneous"


def save_with_external_legend(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def line_panel_by_chip(df, metrics, title, path):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10.8, 2.7 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    chips = sorted(df["chip"].dropna().astype(str).unique())
    for ax, (col, label) in zip(axes, metrics):
        for chip in chips:
            sub = df[df["chip"].astype(str) == chip].sort_values("div")
            if sub.empty:
                continue
            grouped = sub.groupby("div", as_index=False)[col].mean()
            ax.plot(grouped["div"], grouped[col], marker="o", linewidth=1.8, label=f"chip {chip}")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("DIV")
    fig.suptitle(title, y=0.995)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.subplots_adjust(top=0.94, bottom=0.12, hspace=0.32)
    save_with_external_legend(fig, path)


def scatter_with_external_legend(df, x, y, title, path):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for chip, sub in df.groupby("chip"):
        ax.scatter(sub[x], sub[y], s=42, alpha=0.85, label=f"chip {chip}")
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_with_external_legend(fig, path)


def grouped_bar(df, group_col, value_cols, title, ylabel, path):
    plot = df.set_index(group_col)[value_cols].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    plot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title(title)
    ax.set_xlabel(group_col.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.subplots_adjust(bottom=0.24)
    save_with_external_legend(fig, path)


def simple_bar(df, x, y, title, ylabel, path, color="#4c78a8"):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    plot = df.copy().iloc[::-1]
    ax.barh(plot[x].astype(str), plot[y], color=color)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel(x.replace("_", " "))
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.36, right=0.98, bottom=0.13)
    save_with_external_legend(fig, path)


def line_grid_by_chip(df, metrics, title, path, ncols=3):
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows), squeeze=False)
    chips = sorted(df["chip"].dropna().astype(str).unique())
    for ax, (col, label) in zip(axes.flat, metrics):
        for chip in chips:
            sub = df[df["chip"].astype(str) == chip].sort_values("div")
            if sub.empty:
                continue
            grouped = sub.groupby("div", as_index=False)[col].mean()
            ax.plot(grouped["div"], grouped[col], marker="o", linewidth=1.8, label=f"chip {chip}")
        ax.set_title(label)
        ax.set_xlabel("DIV")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="chip", loc="lower center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(title, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.20, wspace=0.28, hspace=0.42)
    save_with_external_legend(fig, path)


def coverage_plot(metrics, strat, experiment_id, path):
    metric_cov = metrics.groupby("div").size().rename("metrics")
    strat_cov = strat.groupby("div").size().rename("strat") if not strat.empty else pd.Series(dtype=float, name="strat")
    coverage = pd.concat([metric_cov, strat_cov], axis=1).fillna(0).reset_index()
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    x = range(len(coverage))
    width = 0.36
    ax.bar([i - width / 2 for i in x], coverage["metrics"], width=width, label="metrics", color="#f28e2b")
    if "strat" in coverage:
        ax.bar([i + width / 2 for i in x], coverage["strat"], width=width, label="strat", color="#59a14f")
    ax.set_title(f"Experiment {experiment_id}: output coverage")
    ax.set_xlabel("DIV")
    ax.set_ylabel("recordings")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(int(v)) if pd.notna(v) else "NA" for v in coverage["div"]])
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.25, left=0.07, right=0.98, top=0.86)
    save_with_external_legend(fig, path)


def correlation_edges_from_counts(counts, threshold=0.2):
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or counts.shape[0] < 2 or counts.shape[1] < 2:
        return 0, np.nan, np.nan
    variances = counts.var(axis=1)
    valid = variances > 0
    if valid.sum() < 2:
        return 0, np.nan, np.nan
    corr = np.corrcoef(counts[valid])
    tri = corr[np.triu_indices_from(corr, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return 0, np.nan, np.nan
    return int((np.abs(tri) > threshold).sum()), float(np.nanmean(np.abs(tri))), float(np.nanmedian(np.abs(tri)))


def build_correlation_bin_sensitivity(experiment_id):
    exp_dir = MARK_ROOT / f"experiment_{experiment_id}"
    conn_dir = exp_dir / "connectvity_neg"
    rows = []
    for counts_path in sorted(conn_dir.glob("*/binned_spike_counts.npy")):
        recording = counts_path.parent.name
        counts_50 = np.load(counts_path)
        n_channels = counts_50.shape[0]
        edges_50, mean_abs_50, median_abs_50 = correlation_edges_from_counts(counts_50)
        rows.append(
            {
                "recording_name": recording,
                "div": div_from_name(recording),
                "chip": chip_from_name(recording),
                "bin_ms": 50,
                "active_channels": n_channels,
                "correlation_edges_abs_gt_0_2": edges_50,
                "mean_abs_correlation": mean_abs_50,
                "median_abs_correlation": median_abs_50,
                "status": "computed_from_saved_50_ms_bins",
            }
        )
        usable = counts_50[:, : (counts_50.shape[1] // 2) * 2]
        counts_100 = usable.reshape(n_channels, -1, 2).sum(axis=2)
        edges_100, mean_abs_100, median_abs_100 = correlation_edges_from_counts(counts_100)
        rows.append(
            {
                "recording_name": recording,
                "div": div_from_name(recording),
                "chip": chip_from_name(recording),
                "bin_ms": 100,
                "active_channels": n_channels,
                "correlation_edges_abs_gt_0_2": edges_100,
                "mean_abs_correlation": mean_abs_100,
                "median_abs_correlation": median_abs_100,
                "status": "computed_by_aggregating_saved_50_ms_bins",
            }
        )
        rows.append(
            {
                "recording_name": recording,
                "div": div_from_name(recording),
                "chip": chip_from_name(recording),
                "bin_ms": 20,
                "active_channels": n_channels,
                "correlation_edges_abs_gt_0_2": np.nan,
                "mean_abs_correlation": np.nan,
                "median_abs_correlation": np.nan,
                "status": "requires_raw_spike_times_not_reconstructable_from_50_ms_bins",
            }
        )
    sensitivity = pd.DataFrame(rows)
    out_csv = conn_dir / "spike_count_correlation_bin_sensitivity.csv"
    sensitivity.to_csv(out_csv, index=False)
    return sensitivity, out_csv


def sensitivity_plot(sensitivity, experiment_id, path):
    computed = sensitivity[sensitivity["correlation_edges_abs_gt_0_2"].notna()].copy()
    if computed.empty:
        return
    summary = computed.groupby(["div", "chip", "bin_ms"], as_index=False)["correlation_edges_abs_gt_0_2"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    for ax, bin_ms in zip(axes, [50, 100]):
        sub_bin = summary[summary["bin_ms"] == bin_ms]
        for chip, sub in sub_bin.groupby("chip"):
            sub = sub.sort_values("div")
            ax.plot(sub["div"], sub["correlation_edges_abs_gt_0_2"], marker="o", linewidth=1.8, label=f"chip {chip}")
        ax.set_title(f"{bin_ms} ms bins")
        ax.set_xlabel("DIV")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("|r| > 0.2 correlation edges")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="chip", loc="lower center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Experiment {experiment_id}: spike-count correlation bin sensitivity", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.24, wspace=0.18)
    save_with_external_legend(fig, path)


def safe_name(name, max_len=90):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


def network_map_panel(recording_dir, recording_name, output_path, top_percentile=95, top_n=120):
    sttc_strength = pd.read_csv(recording_dir / "sttc_node_strength.csv")
    corr_strength = pd.read_csv(recording_dir / "correlation_node_strength.csv")
    sttc_edges = pd.read_csv(recording_dir / "sttc_edges.csv")

    sttc_grid = ChannelValuesToGrid(sttc_strength)
    corr_grid = ChannelValuesToGrid(corr_strength)
    selected_edges = SelectStrongEdges(sttc_edges, TopPercentile=top_percentile, TopN=top_n)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3))
    for ax, grid, title, cmap in [
        (axes[0], sttc_grid, "STTC node strength", "magma"),
        (axes[1], corr_grid, "Correlation node strength", "viridis"),
    ]:
        image = ax.imshow(grid, cmap=cmap, origin="upper", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.set_xlim(-0.5, 63.5)
        ax.set_ylim(63.5, -0.5)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    node_rows = []
    for row in sttc_strength.itertuples(index=False):
        grid_row, grid_col = ChannelToGrid(row.channel)
        node_rows.append({"row": grid_row, "col": grid_col, "strength": row.strength})
    nodes = pd.DataFrame(node_rows)
    axes[2].scatter(nodes["col"], nodes["row"], c=nodes["strength"], cmap="magma", s=18, alpha=0.9, linewidths=0)

    if not selected_edges.empty:
        scores = selected_edges["score"].to_numpy(dtype=float)
        score_min = np.nanmin(scores)
        score_max = np.nanmax(scores)
        for edge in selected_edges.itertuples(index=False):
            source_row, source_col = ChannelToGrid(edge.source)
            target_row, target_col = ChannelToGrid(edge.target)
            if score_max > score_min:
                linewidth = 0.4 + 2.2 * (edge.score - score_min) / (score_max - score_min)
            else:
                linewidth = 1.4
            axes[2].plot([source_col, target_col], [source_row, target_row], color="#d62728", alpha=0.34, linewidth=linewidth)

    axes[2].set_title(f"Top STTC network edges ({len(selected_edges)})")
    axes[2].set_xlabel("column")
    axes[2].set_ylabel("row")
    axes[2].set_xlim(-0.5, 63.5)
    axes[2].set_ylim(63.5, -0.5)
    axes[2].set_aspect("equal")
    axes[2].grid(True, color="#d0d0d0", linewidth=0.25, alpha=0.35)

    fig.suptitle(recording_name, y=0.99)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.86, bottom=0.10, wspace=0.22)
    save_with_external_legend(fig, output_path)
    return selected_edges


def build_network_maps(experiment_id, fig_dir, max_recordings=3):
    exp_dir = MARK_ROOT / f"experiment_{experiment_id}"
    conn_dir = exp_dir / "connectvity_neg"
    summary = pd.read_csv(conn_dir / "connectivity_summary.csv")
    selected = summary.sort_values("sttc_edges", ascending=False).head(max_recordings).copy()
    map_dir = fig_dir / "network_maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in selected.itertuples(index=False):
        recording_name = record.recording_name
        recording_dir = conn_dir / recording_name
        required = [
            recording_dir / "sttc_node_strength.csv",
            recording_dir / "correlation_node_strength.csv",
            recording_dir / "sttc_edges.csv",
        ]
        if not all(path.exists() for path in required):
            continue
        output_path = map_dir / f"{safe_name(recording_name)}_network_map.png"
        selected_edges = network_map_panel(recording_dir, recording_name, output_path)
        rows.append(
            {
                "recording_name": recording_name,
                "div": div_from_name(recording_name),
                "chip": chip_from_name(recording_name),
                "active_channels": record.active_channels,
                "active_negative_spikes": record.active_negative_spikes,
                "sttc_edges": record.sttc_edges,
                "correlation_edges": record.correlation_edges,
                "shown_top_sttc_edges": len(selected_edges),
                "network_map": str(output_path),
            }
        )

    out_csv = map_dir / "selected_network_maps.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


def classification_campaigns(exp_dir, include_mamba=False):
    suffix = "_mamba" if include_mamba else ""
    return [
        ("DIV target", exp_dir / f"recording_classification_negative_spikes_div_ml_campaign{suffix}"),
        ("Culture-phase target", exp_dir / f"recording_classification_negative_spikes_culture_phase_ml_campaign{suffix}"),
    ]


def div_regression_campaign(exp_dir):
    return exp_dir / "recording_regression_negative_spikes_div_ml_campaign"


def regression_summary(exp_dir):
    campaign_dir = div_regression_campaign(exp_dir)
    ranking_path = campaign_dir / "algorithm_comparison.csv"
    summary_path = campaign_dir / "summary_metrics.csv"
    source = ranking_path if ranking_path.exists() else summary_path
    if not source.exists():
        return campaign_dir, pd.DataFrame(), {}
    results = pd.read_csv(source)
    if "rank" not in results.columns:
        results = results.sort_values("mean_test_mae_days").reset_index(drop=True)
        results.insert(0, "rank", np.arange(1, len(results) + 1))
    metadata_path = campaign_dir / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return campaign_dir, results, metadata


def regression_interpretation(results):
    if results.empty:
        return "No DIV regression results were available."
    best = results.sort_values("mean_test_mae_days").iloc[0]
    return (
        f"The best cross-validated model was {best['model']}, with mean absolute error "
        f"{best['mean_test_mae_days']:.2f} days, RMSE {best['mean_test_rmse_days']:.2f} days, "
        f"R2 {best['mean_test_r2']:.3f}, and {100 * best['mean_test_within_10_days']:.1f} percent "
        "of predictions within 10 days. Results should be interpreted as internal grouped "
        "cross-validation estimates rather than external validation."
    )


def mamba_comparison_table(exp_dir):
    rows = []
    for target_label, mamba_dir in classification_campaigns(exp_dir, include_mamba=True):
        mamba_summary = mamba_dir / "summary_metrics.csv"
        if not mamba_summary.exists():
            continue

        mamba_df = pd.read_csv(mamba_summary)
        if mamba_df.empty or "waveform_sequence_mamba" not in set(mamba_df["model"]):
            continue

        non_mamba_df = mamba_df[mamba_df["model"] != "waveform_sequence_mamba"].copy()
        mamba_row = mamba_df[mamba_df["model"] == "waveform_sequence_mamba"].iloc[0]
        best_same_run = non_mamba_df.sort_values("mean_test_balanced_accuracy", ascending=False).iloc[0] if not non_mamba_df.empty else None

        rows.append(
            {
                "target": target_label,
                "mamba_bal_acc": mamba_row["mean_test_balanced_accuracy"],
                "mamba_std": mamba_row["std_test_balanced_accuracy"],
                "mamba_train": mamba_row["mean_train_balanced_accuracy"],
                "best_classical": best_same_run["model"] if best_same_run is not None else "NA",
                "best_classical_bal_acc": best_same_run["mean_test_balanced_accuracy"] if best_same_run is not None else np.nan,
                "delta_mamba": mamba_row["mean_test_balanced_accuracy"] - best_same_run["mean_test_balanced_accuracy"] if best_same_run is not None else np.nan,
                "folds": mamba_row["n_folds"],
            }
        )
    return pd.DataFrame(rows)


def mamba_comparison_plot(exp_dir, fig_dir):
    comparison = mamba_comparison_table(exp_dir)
    if comparison.empty:
        return comparison

    labels = comparison["target"].to_list()
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.bar(x - width / 2, comparison["mamba_bal_acc"], width=width, label="Mamba waveform sequence", color="#4c78a8")
    ax.bar(x + width / 2, comparison["best_classical_bal_acc"], width=width, label="Best classical", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(0.35, float(np.nanmax(comparison[["mamba_bal_acc", "best_classical_bal_acc"]].to_numpy())) + 0.06))
    ax.set_ylabel("mean test balanced accuracy")
    ax.set_title("Mamba versus classical recording-level classifiers")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.25)
    save_with_external_legend(fig, fig_dir / "mamba_vs_classical_balanced_accuracy.png")
    comparison.to_csv(fig_dir / "mamba_vs_classical_summary.csv", index=False)
    return comparison


def mamba_interpretation(comparison):
    parts = []
    for row in comparison.itertuples(index=False):
        relation = "below" if row.delta_mamba < 0 else "above"
        parts.append(
            f"For {row.target}, Mamba reached balanced accuracy {row.mamba_bal_acc:.3f} "
            f"versus {row.best_classical_bal_acc:.3f} for the best classical model in the same run "
            f"({row.best_classical}), i.e. {abs(row.delta_mamba):.3f} {relation} that reference."
        )
    return " ".join(parts)


def build_mark_dataset(exp_dir):
    metrics = pd.read_csv(exp_dir / "metrics" / "all_recordings_spike_burst_summary.csv")
    metrics["recording_name"] = metrics["RecordingName"].astype(str)
    metrics["div"] = metrics["recording_name"].map(div_from_name)
    metrics["chip"] = metrics["recording_name"].map(chip_from_name)

    conn = pd.read_csv(exp_dir / "connectvity_neg" / "connectivity_summary.csv")
    conn["div"] = conn["recording_name"].map(div_from_name)
    conn["chip"] = conn["recording_name"].map(chip_from_name)
    possible_edges = conn["active_channels"] * (conn["active_channels"] - 1) / 2
    conn["sttc_density_percent"] = (conn["sttc_edges"] / possible_edges.replace(0, pd.NA)) * 100

    run_infos = []
    for path in sorted((exp_dir / "spike_stratification_neg").glob("*/run_info.csv")):
        one = pd.read_csv(path)
        one["div"] = one["recording_name"].map(div_from_name)
        one["chip"] = one["recording_name"].map(chip_from_name)
        run_infos.append(one)
    strat = pd.concat(run_infos, ignore_index=True) if run_infos else pd.DataFrame()
    return metrics, conn, strat


def build_mark_figures(experiment_id):
    exp_dir = MARK_ROOT / f"experiment_{experiment_id}"
    fig_dir = exp_dir / f"experiment_{experiment_id}_legend_fixed_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics, conn, strat = build_mark_dataset(exp_dir)
    line_panel_by_chip(
        metrics,
        [
            ("ActiveElectrodesNumber", "active electrodes"),
            ("NSpikesWell", "well spikes"),
            ("SpikesWellRate", "well firing rate"),
            ("NBurstsWell", "well bursts"),
            ("BurstsSpikesPercentageWell", "burst spike percentage"),
        ],
        f"Experiment {experiment_id}: spike/burst metrics by DIV and chip",
        fig_dir / "metrics_by_div_chip.png",
    )
    line_panel_by_chip(
        conn,
        [
            ("active_channels", "active channels"),
            ("active_negative_spikes", "active negative spikes"),
            ("sttc_edges", "STTC edges"),
            ("correlation_edges", "correlation edges"),
        ],
        f"Experiment {experiment_id}: connectivity by DIV and chip",
        fig_dir / "connectivity_edges_by_div_chip.png",
    )
    scatter_with_external_legend(
        conn,
        "active_channels",
        "sttc_edges",
        f"Experiment {experiment_id}: STTC edges versus active channels",
        fig_dir / "sttc_edges_vs_active_channels.png",
    )
    if not strat.empty:
        line_panel_by_chip(
            strat,
            [
                ("n_input_spikes", "input spikes"),
                ("n_after_waveform_filter", "plausible negative spikes"),
                ("n_after_active_filter", "active negative spikes"),
                ("n_spikes", "sampled spikes"),
                ("n_active_channels", "active channels"),
            ],
            f"Experiment {experiment_id}: stratification filtering by DIV and chip",
            fig_dir / "stratification_filtering.png",
        )

    div_cls = exp_dir / "recording_classification_negative_spikes_div_ml_campaign" / "summary_metrics.csv"
    if div_cls.exists():
        cls = pd.read_csv(div_cls).sort_values("mean_test_balanced_accuracy", ascending=False).head(10)
        simple_bar(
            cls,
            "model",
            "mean_test_balanced_accuracy",
            f"Experiment {experiment_id}: top DIV classifiers",
            "mean test balanced accuracy",
            fig_dir / "classification_top_div.png",
            color="#59a14f",
        )
    phase_cls = exp_dir / "recording_classification_negative_spikes_culture_phase_ml_campaign" / "summary_metrics.csv"
    if phase_cls.exists():
        cls = pd.read_csv(phase_cls).sort_values("mean_test_balanced_accuracy", ascending=False).head(10)
        simple_bar(
            cls,
            "model",
            "mean_test_balanced_accuracy",
            f"Experiment {experiment_id}: top culture-phase classifiers",
            "mean test balanced accuracy",
            fig_dir / "classification_top_culture_phase.png",
            color="#b07aa1",
        )
    return fig_dir


def build_mark_very_detailed_figures(experiment_id):
    exp_dir = MARK_ROOT / f"experiment_{experiment_id}"
    fig_dir = exp_dir / f"experiment_{experiment_id}_very_detailed_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics, conn, strat = build_mark_dataset(exp_dir)

    line_grid_by_chip(
        metrics,
        [
            ("NSpikesWell", "NSpikesWell"),
            ("SpikesWellRate", "SpikesWellRate"),
            ("BurstsSpikesPercentageWell", "Burst spike %"),
        ],
        f"Experiment {experiment_id}: spike/burst trends",
        fig_dir / "metrics_trends.png",
    )
    line_grid_by_chip(
        conn,
        [
            ("active_negative_spikes", "Active negative spikes"),
            ("sttc_edges", "STTC edges"),
            ("sttc_density_percent", "STTC density %"),
        ],
        f"Experiment {experiment_id}: connectivity trends",
        fig_dir / "connectivity_trends.png",
    )
    coverage_plot(metrics, strat, experiment_id, fig_dir / "coverage.png")
    if not strat.empty:
        totals = pd.DataFrame(
            {
                "stage": ["input", "label neg", "waveform neg", "active ch", "sampled"],
                "spikes": [
                    strat["n_input_spikes"].sum(),
                    strat["n_after_label_filter"].sum(),
                    strat["n_after_waveform_filter"].sum(),
                    strat["n_after_active_filter"].sum(),
                    strat["n_spikes"].sum(),
                ],
            }
        )
        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        bars = ax.bar(totals["stage"], totals["spikes"], color=["#6b7280", "#4e79a7", "#59a14f", "#f28e2b", "#b07aa1"])
        for bar, value in zip(bars, totals["spikes"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value / 1000:,.0f}k", ha="center", va="bottom", fontsize=10)
        ax.set_title("Stratification filtering")
        ax.set_ylabel("spikes")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(bottom=0.15, left=0.07, right=0.98, top=0.86)
        save_with_external_legend(fig, fig_dir / "strat_filtering.png")
    sensitivity, _ = build_correlation_bin_sensitivity(experiment_id)
    sensitivity_plot(sensitivity, experiment_id, fig_dir / "correlation_bin_sensitivity.png")
    build_network_maps(experiment_id, fig_dir)
    mamba_comparison_plot(exp_dir, fig_dir)
    return fig_dir


def build_3brain_figures():
    fig_dir = THREEBRAIN_ROOT / "dataset_26-06_negative_integrated_report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    strat = pd.read_csv(THREEBRAIN_ROOT / "spike_stratification_negative_waveforms" / "26-06" / "per_recording_summary.csv")
    strat["condition"] = strat["recording_name"].map(condition_3brain)
    conn = pd.read_csv(THREEBRAIN_ROOT / "connectivity_negative_spikes" / "26-06" / "connectivity_summary.csv")
    conn["condition"] = conn["recording_name"].map(condition_3brain)
    cls = pd.read_csv(THREEBRAIN_ROOT / "spike_classification_negative_waveforms" / "26-06_recording_level" / "summary_metrics.csv")
    clusters = pd.read_csv(THREEBRAIN_ROOT / "dataset_26-06_negative_integrated_report" / "global_cluster_profile_by_label.csv")

    active = strat.groupby("condition", as_index=False)["n_after_active_filter"].sum()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(active["condition"], active["n_after_active_filter"], color="#4c78a8")
    ax.set_title("Active plausible negative spikes by condition")
    ax.set_xlabel("condition")
    ax.set_ylabel("active negative spikes")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save_with_external_legend(fig, fig_dir / "active_negative_spikes_by_condition.png")

    grouped_bar(
        cls.sort_values("mean_test_balanced_accuracy", ascending=False),
        "model",
        ["mean_test_balanced_accuracy", "mean_train_balanced_accuracy"],
        "Recording-level classification balanced accuracy",
        "balanced accuracy",
        fig_dir / "classification_balanced_accuracy.png",
    )
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")

    conn_summary = conn.groupby("condition", as_index=False)[["sttc_edges", "correlation_edges"]].mean()
    grouped_bar(conn_summary, "condition", ["sttc_edges", "correlation_edges"], "Mean connectivity edges by condition", "edge count", fig_dir / "connectivity_edges_by_label.png")

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for condition, sub in conn.groupby("condition"):
        ax.scatter(sub["active_negative_spikes"], sub["sttc_edges"], s=44, alpha=0.85, label=condition)
    ax.set_title("STTC edges versus active negative spikes")
    ax.set_xlabel("active negative spikes")
    ax.set_ylabel("STTC edges")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_with_external_legend(fig, fig_dir / "connectivity_spikes_vs_sttc_edges.png")

    cluster_cols = [c for c in clusters.columns if c not in {"label_normalized", "condition", "recordings"}]
    label_col = "label_normalized" if "label_normalized" in clusters.columns else clusters.columns[0]
    grouped_bar(clusters, label_col, cluster_cols, "Global cluster profile by condition", "cluster fraction", fig_dir / "stratification_global_cluster_profile_by_label.png")
    return fig_dir


def pdf_appendix(experiment_id, fig_dir):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22))
    styles.add(ParagraphStyle("BodySmall", parent=styles["BodyText"], fontSize=9, leading=11))
    out = MARK_ROOT / f"experiment_{experiment_id}" / f"experiment_{experiment_id}_legend_fixed_figure_appendix.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=landscape(A4), leftMargin=1.0 * cm, rightMargin=1.0 * cm, topMargin=0.9 * cm, bottomMargin=0.9 * cm)
    story = [
        Paragraph(f"Experiment {experiment_id}: Legend-Fixed Figure Appendix", styles["TitleCenter"]),
        Paragraph("The plots below are regenerated from the saved CSV outputs only. Numerical analysis results are unchanged; legends were moved outside the plotting area to avoid overlap.", styles["BodySmall"]),
        Spacer(1, 0.25 * cm),
    ]
    for path in sorted(fig_dir.glob("*.png")):
        img = Image(str(path))
        max_w, max_h = 25.0 * cm, 13.0 * cm
        scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        story.append(img)
        story.append(Paragraph(path.name.replace("_", " ").replace(".png", ""), styles["BodySmall"]))
        story.append(Spacer(1, 0.35 * cm))
    doc.build(story)
    return out


def fmt(value):
    if pd.isna(value):
        return "NA"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:,.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def fmt_day(value):
    if pd.isna(value):
        return "NA"
    return str(int(round(float(value))))


def paragraph(text, styles, style="BodySmall"):
    return Paragraph(str(text), styles[style])


def table_from_df(df, styles, max_rows=20):
    df = df.head(max_rows).copy()
    header_style = "HeaderCell" if hasattr(styles, "byName") and "HeaderCell" in styles.byName else "Cell"
    data = [[paragraph(col.replace("_", " "), styles, header_style) for col in df.columns]]
    for _, row in df.iterrows():
        data.append([paragraph(fmt(row[col]), styles, "Cell") for col in df.columns])
    available = 27.0 * cm
    col_width = available / max(1, len(df.columns))
    table = Table(data, colWidths=[col_width] * len(df.columns), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8dc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8f9")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def numeric_summary(df, columns):
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": col,
                "sum": values.sum(),
                "mean": values.mean(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def add_image(story, path, styles):
    if not Path(path).exists():
        return
    img = Image(str(path))
    max_w, max_h = 25.8 * cm, 11.5 * cm
    scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    story.append(img)
    story.append(paragraph(Path(path).name.replace("_", " ").replace(".png", ""), styles, "Cell"))
    story.append(Spacer(1, 0.25 * cm))


def build_mark_full_report(experiment_id, fig_dir):
    exp_dir = MARK_ROOT / f"experiment_{experiment_id}"
    metrics, conn, strat = build_mark_dataset(exp_dir)
    out = exp_dir / f"experiment_{experiment_id}_legend_fixed_full_report.pdf"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22))
    styles.add(ParagraphStyle("Section", parent=styles["Heading1"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle("BodySmall", parent=styles["BodyText"], fontSize=8.7, leading=10.5, spaceAfter=4))
    styles.add(ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=6.2, leading=7.2))
    styles.add(ParagraphStyle("HeaderCell", parent=styles["Cell"], textColor=colors.white))

    story = [
        paragraph(f"Experiment {experiment_id}: Detailed MEA Analysis Report", styles, "TitleCenter"),
        paragraph(
            "This legend-fixed report is regenerated from the saved Mark output CSV files. "
            "The numerical outputs are unchanged; summary figures were redrawn with legends outside the plotting area.",
            styles,
        ),
        Spacer(1, 0.2 * cm),
    ]

    n_recordings = len(metrics)
    chips = ", ".join(sorted(metrics["chip"].dropna().astype(str).unique()))
    divs = pd.to_numeric(metrics["div"], errors="coerce").dropna()
    story.append(paragraph("Dataset", styles, "Section"))
    story.append(
        paragraph(
            f"The dataset contains {n_recordings} recordings. Detected chips are {chips}. "
            f"The DIV range inferred from recording names is {int(divs.min())}-{int(divs.max())}."
            if not divs.empty
            else f"The dataset contains {n_recordings} recordings. Detected chips are {chips}.",
            styles,
        )
    )

    story.append(paragraph("Spike And Burst Metrics", styles, "Section"))
    story.append(
        table_from_df(
            numeric_summary(
                metrics,
                [
                    "ActiveElectrodesNumber",
                    "NSpikesWell",
                    "SpikesWellRate",
                    "ISIWellAverage",
                    "PeakToPeakAverageWell",
                    "NBurstsWell",
                    "BurstsRateWell",
                    "BurstsDurationWellAverage",
                    "BurstsSpikesPercentageWell",
                ],
            ),
            styles,
            max_rows=20,
        )
    )
    add_image(story, fig_dir / "metrics_by_div_chip.png", styles)

    story.append(paragraph("Negative-Spike Stratification", styles, "Section"))
    if not strat.empty:
        story.append(
            paragraph(
                "Stratification uses biologically plausible negative spikes: detector-label filtering, waveform-polarity filtering, "
                "active-channel filtering at 0.25 Hz, PCA retaining 90 percent variance, and K-means with Euclidean distance.",
                styles,
            )
        )
        story.append(
            table_from_df(
                numeric_summary(
                    strat,
                    [
                        "n_input_spikes",
                        "n_after_label_filter",
                        "n_after_waveform_filter",
                        "n_after_active_filter",
                        "n_spikes",
                        "n_active_channels",
                        "n_clusters",
                    ],
                ),
                styles,
                max_rows=20,
            )
        )
        add_image(story, fig_dir / "stratification_filtering.png", styles)

    story.append(paragraph("Recording-Level Classification", styles, "Section"))
    for label, path in [
        ("DIV target", exp_dir / "recording_classification_negative_spikes_div_ml_campaign" / "summary_metrics.csv"),
        ("Culture-phase target", exp_dir / "recording_classification_negative_spikes_culture_phase_ml_campaign" / "summary_metrics.csv"),
    ]:
        if path.exists():
            cls = pd.read_csv(path).sort_values("mean_test_balanced_accuracy", ascending=False)
            story.append(paragraph(label, styles, "Section"))
            story.append(table_from_df(cls[["model", "mean_test_accuracy", "mean_test_balanced_accuracy", "std_test_balanced_accuracy", "mean_train_balanced_accuracy", "mean_pca_components", "n_folds"]], styles, max_rows=12))
    add_image(story, fig_dir / "classification_top_div.png", styles)
    add_image(story, fig_dir / "classification_top_culture_phase.png", styles)

    story.append(PageBreak())
    story.append(paragraph("Connectivity", styles, "Section"))
    story.append(
        paragraph(
            "Connectivity was summarized from the negative-spike connectivity outputs. The report includes active channels, active negative spikes, STTC edges, and spike-count correlation edges.",
            styles,
        )
    )
    story.append(
        table_from_df(
            numeric_summary(conn, ["active_channels", "active_negative_spikes", "sttc_edges", "correlation_edges", "elapsed_seconds"]),
            styles,
            max_rows=20,
        )
    )
    add_image(story, fig_dir / "connectivity_edges_by_div_chip.png", styles)
    add_image(story, fig_dir / "sttc_edges_vs_active_channels.png", styles)

    story.append(paragraph("Per-Recording Tables", styles, "Section"))
    story.append(paragraph("Spike/burst metrics, first 30 recordings.", styles))
    story.append(table_from_df(metrics[["recording_name", "div", "chip", "ActiveElectrodesNumber", "NSpikesWell", "SpikesWellRate", "NBurstsWell", "BurstsSpikesPercentageWell"]].sort_values(["div", "chip"]), styles, max_rows=30))
    story.append(Spacer(1, 0.2 * cm))
    story.append(paragraph("Connectivity metrics, first 30 recordings.", styles))
    story.append(table_from_df(conn[["recording_name", "div", "chip", "active_channels", "active_negative_spikes", "sttc_edges", "correlation_edges"]].sort_values(["div", "chip"]), styles, max_rows=30))

    doc = SimpleDocTemplate(str(out), pagesize=landscape(A4), leftMargin=1.0 * cm, rightMargin=1.0 * cm, topMargin=0.9 * cm, bottomMargin=0.9 * cm)
    doc.build(story)
    return out


def build_mark_very_detailed_report(experiment_id, fig_dir):
    exp_dir = MARK_ROOT / f"experiment_{experiment_id}"
    metrics, conn, strat = build_mark_dataset(exp_dir)
    out = exp_dir / f"experiment_{experiment_id}_very_detailed_with_methods_report_legend_fixed.pdf"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22))
    styles.add(ParagraphStyle("Section", parent=styles["Heading1"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle("BodySmall", parent=styles["BodyText"], fontSize=8.7, leading=10.5, spaceAfter=4))
    styles.add(ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=6.2, leading=7.2))
    styles.add(ParagraphStyle("HeaderCell", parent=styles["Cell"], textColor=colors.white))

    divs = pd.to_numeric(metrics["div"], errors="coerce").dropna()
    chips = ", ".join(sorted(metrics["chip"].dropna().astype(str).unique()))
    story = [
        paragraph(f"Experiment {experiment_id}: Very Detailed MEA Report With Methods", styles, "TitleCenter"),
        paragraph(
            "Legend-fixed version regenerated from the saved Mark CSV outputs. Numerical results are unchanged. "
            "Summary figures were redrawn with legends outside the plotting area.",
            styles,
        ),
        Spacer(1, 0.18 * cm),
        paragraph("Dataset Description", styles, "Section"),
        paragraph(
            f"The experiment includes {len(metrics)} recordings across chips {chips}. "
            f"DIV values inferred from recording names span {int(divs.min())}-{int(divs.max())}."
            if not divs.empty
            else f"The experiment includes {len(metrics)} recordings across chips {chips}.",
            styles,
        ),
        paragraph("Methods", styles, "Section"),
        paragraph(
            "Spike/burst metrics were summarized at recording level from the processed MEA outputs. "
            "Negative-spike stratification used detector-label filtering, waveform polarity filtering, active-channel filtering at 0.25 Hz, PCA retaining 90 percent variance, and K-means with Euclidean distance. "
            "Connectivity was summarized from active plausible negative spikes using STTC and spike-count correlation edge counts. "
            "Recording-level classification summaries report cross-validated model performance from the saved classification campaigns. "
            "DIV regression predicts culture age in days from recording-level activity descriptors, waveform representations, or their combination, using grouped cross-validation by biological unit.",
            styles,
        ),
        paragraph("Spike And Burst Metrics", styles, "Section"),
        table_from_df(
            numeric_summary(
                metrics,
                [
                    "ActiveElectrodesNumber",
                    "NSpikesWell",
                    "SpikesWellRate",
                    "ISIWellAverage",
                    "PeakToPeakAverageWell",
                    "NBurstsWell",
                    "BurstsRateWell",
                    "BurstsDurationWellAverage",
                    "BurstsSpikesPercentageWell",
                ],
            ),
            styles,
        ),
    ]
    add_image(story, fig_dir / "metrics_trends.png", styles)

    story.append(paragraph("Stratification", styles, "Section"))
    if not strat.empty:
        story.append(
            table_from_df(
                numeric_summary(
                    strat,
                    [
                        "n_input_spikes",
                        "n_after_label_filter",
                        "n_after_waveform_filter",
                        "n_after_active_filter",
                        "n_spikes",
                        "n_active_channels",
                        "n_clusters",
                    ],
                ),
                styles,
            )
        )
        add_image(story, fig_dir / "strat_filtering.png", styles)
    add_image(story, fig_dir / "coverage.png", styles)

    story.append(PageBreak())
    story.append(paragraph("Classification", styles, "Section"))
    story.append(
        paragraph(
            "The classical recording-level classifiers use scalar activity descriptors, waveform summary statistics, and MiniRocket waveform features aggregated at recording level. "
            "The new Mamba runs add a sequence model that receives the waveform profile as an ordered temporal object rather than as only a fixed vector of engineered features.",
            styles,
        )
    )
    for label, campaign_dir in classification_campaigns(exp_dir, include_mamba=False):
        path = campaign_dir / "summary_metrics.csv"
        if path.exists():
            cls = pd.read_csv(path).sort_values("mean_test_balanced_accuracy", ascending=False)
            story.append(paragraph(label, styles, "Section"))
            story.append(table_from_df(cls[["model", "mean_test_accuracy", "mean_test_balanced_accuracy", "std_test_balanced_accuracy", "mean_train_balanced_accuracy", "mean_pca_components", "n_folds"]], styles, max_rows=15))

    mamba_comparison = mamba_comparison_table(exp_dir)
    if not mamba_comparison.empty:
        story.append(paragraph("Mamba Sequence Classifier", styles, "Section"))
        story.append(
            paragraph(
                "Mamba is a selective state-space sequence model. In contrast to self-attention, whose memory and compute grow quadratically with sequence length, Mamba uses a recurrent state-space formulation with linear scaling in sequence length. "
                "Its key idea is selectivity: the state update is conditioned on the current input, so the model can decide which temporal information to keep, suppress, or propagate. "
                "This makes it suitable for waveform-level MEA classification, where the order of samples around the spike trough carries information about spike shape, recovery phase, and waveform dispersion.",
                styles,
            )
        )
        story.append(
            paragraph(
                "In these runs, each recording is represented as a five-channel waveform sequence built from the negative-spike waveforms: mean, standard deviation, median, 25th percentile, and 75th percentile across sampled spikes at each time point. "
                "The comparison is still recording-level and uses the same grouped cross-validation logic as the classical models, keeping biological units disjoint between train and test folds. "
                "The Mamba model is therefore tested on the same prediction task, but with a sequence-preserving waveform representation.",
                styles,
            )
        )
        story.append(
            paragraph(
                "Reference: Gu and Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces, arXiv:2312.00752, 2023/2024. "
                "The implementation in this repository uses the local `src/brain_3d/Mamba.py` module, with a portable PyTorch selective state-space backend unless the official `mamba_ssm` backend is explicitly enabled.",
                styles,
            )
        )
        story.append(table_from_df(mamba_comparison, styles, max_rows=10))
        story.append(paragraph(mamba_interpretation(mamba_comparison), styles))
        add_image(story, fig_dir / "mamba_vs_classical_balanced_accuracy.png", styles)

        for label, mamba_dir in classification_campaigns(exp_dir, include_mamba=True):
            summary_path = mamba_dir / "summary_metrics.csv"
            if not summary_path.exists():
                continue
            mamba_cls = pd.read_csv(summary_path).sort_values("mean_test_balanced_accuracy", ascending=False)
            if mamba_cls.empty:
                continue
            story.append(paragraph(f"{label}: Mamba Run Ranking", styles, "Section"))
            story.append(
                table_from_df(
                    mamba_cls[
                        [
                            "model",
                            "mean_test_accuracy",
                            "mean_test_balanced_accuracy",
                            "std_test_balanced_accuracy",
                            "mean_train_balanced_accuracy",
                            "mean_pca_components",
                            "n_folds",
                        ]
                    ],
                    styles,
                    max_rows=18,
                )
            )
            add_image(story, mamba_dir / "confusion_matrix_waveform_sequence_mamba.png", styles)

    regression_dir, regression, regression_meta = regression_summary(exp_dir)
    if not regression.empty:
        story.append(PageBreak())
        story.append(paragraph("DIV Regression", styles, "Section"))
        story.append(
            paragraph(
                "The continuous target is days in vitro (DIV). Each sample is one recording, and train-test separation is performed by biological_unit_id so recordings from the same biological unit do not appear in both partitions. "
                "The campaign compares three input families: activity descriptors, waveform features derived with MiniRocket, and their combined representation. "
                "The evaluated regressors are a mean-prediction dummy baseline, Ridge, Elastic Net, linear SVR, RBF SVR, Random Forest, and Extra Trees. "
                "For linear and kernel models, PCA is fitted inside each training fold and retains 90 percent of explained variance; tree ensembles use the original feature representation. "
                "Performance is aggregated across repeated grouped folds using MAE and RMSE in days, R2, Spearman and Pearson association, and the proportions of predictions within 5, 10, and 15 days.",
                styles,
            )
        )
        if regression_meta:
            params = regression_meta.get("training_parameters", {})
            story.append(
                paragraph(
                    f"The saved campaign contains {regression_meta.get('n_recordings', 'NA')} recordings from "
                    f"{regression_meta.get('n_biological_units', 'NA')} biological units over DIV "
                    f"{fmt_day(regression_meta.get('div_min', np.nan))}-{fmt_day(regression_meta.get('div_max', np.nan))}. "
                    f"The configuration requested up to {params.get('n_splits', 'NA')} grouped splits repeated "
                    f"{params.get('n_repeats', 'NA')} times; the effective number of folds is limited by the number "
                    f"of biological units, giving {int(regression['n_folds'].max())} reported fold evaluations per model. "
                    f"The run used a maximum of "
                    f"{params.get('max_spikes_per_recording', 'NA')} spikes per recording, and "
                    f"{params.get('num_kernels', 'NA')} MiniRocket kernels. Negative plausible spikes were selected "
                    "using detector labels, waveform polarity, a central-extremum constraint, and the 0.25 Hz active-channel threshold.",
                    styles,
                )
            )
        ranking_columns = [
            "rank",
            "model",
            "mean_test_mae_days",
            "std_test_mae_days",
            "mean_test_rmse_days",
            "mean_test_r2",
            "mean_test_spearman",
            "mean_test_within_10_days",
            "n_folds",
        ]
        story.append(paragraph("Top Cross-Validated Models", styles, "Section"))
        story.append(table_from_df(regression[ranking_columns], styles, max_rows=12))
        story.append(paragraph(regression_interpretation(regression), styles))
        add_image(story, regression_dir / "model_ranking_mae.png", styles)
        add_image(story, regression_dir / "best_model_true_vs_predicted_div.png", styles)

    story.append(paragraph("Connectivity", styles, "Section"))
    story.append(
        table_from_df(
            numeric_summary(conn, ["active_channels", "active_negative_spikes", "sttc_edges", "correlation_edges", "elapsed_seconds"]),
            styles,
        )
    )
    add_image(story, fig_dir / "connectivity_trends.png", styles)

    network_maps_csv = fig_dir / "network_maps" / "selected_network_maps.csv"
    if network_maps_csv.exists():
        network_maps = pd.read_csv(network_maps_csv)
        if not network_maps.empty:
            story.append(paragraph("Spatial Network Maps", styles, "Section"))
            story.append(
                paragraph(
                    "The strongest functional networks were visualized on the physical 64 x 64 MEA grid. "
                    "For each experiment, the report shows the recordings with the largest number of STTC edges. "
                    "Each panel contains the STTC node-strength map, the spike-count-correlation node-strength map, and the strongest STTC edges drawn between electrode coordinates. "
                    "Edges are restricted to the upper tail of the STTC edge-weight distribution, capped at 120 edges, so the spatial network remains readable.",
                    styles,
                )
            )
            story.append(
                table_from_df(
                    network_maps[
                        [
                            "recording_name",
                            "div",
                            "chip",
                            "active_channels",
                            "active_negative_spikes",
                            "sttc_edges",
                            "correlation_edges",
                            "shown_top_sttc_edges",
                        ]
                    ],
                    styles,
                    max_rows=5,
                )
            )
            for map_path in network_maps["network_map"].head(3):
                add_image(story, Path(map_path), styles)

    story.append(paragraph("Spike-Count Correlation Bin Sensitivity", styles, "Section"))
    sensitivity_csv = exp_dir / "connectvity_neg" / "spike_count_correlation_bin_sensitivity.csv"
    if sensitivity_csv.exists():
        sensitivity = pd.read_csv(sensitivity_csv)
        computed = sensitivity[sensitivity["correlation_edges_abs_gt_0_2"].notna()].copy()
        story.append(
            paragraph(
                "The original spike-count correlation network used non-overlapping 50 ms bins and an absolute correlation threshold of 0.2. "
                "This choice follows published MEA functional-connectivity practice in which spike trains are converted into 50 ms spike-count vectors and Pearson correlations are computed between channel pairs. "
                "Because binned spike-count correlation depends on the selected temporal window, a sensitivity check was added. "
                "Using the saved 50 ms binned count matrices, the report recomputes correlation-edge counts at 50 ms and at 100 ms by summing adjacent 50 ms bins. "
                "A 20 ms sensitivity point is listed as recommended but not computed from these saved outputs, because a finer bin cannot be reconstructed from already aggregated 50 ms counts; it requires rerunning only the connectivity step from raw spike times.",
                styles,
            )
        )
        story.append(
            paragraph(
                "References for this methodological choice and caveat: Frontiers in Network Physiology 2024 used non-overlapping 50 ms MEA spike-count windows for Pearson functional-connectivity estimates (https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2024.1441345/full); Cutts and Eglen 2014 describe spike-count correlation as Pearson correlation on binned spike trains and emphasize dependence on the time window, motivating complementary use of STTC (https://pmc.ncbi.nlm.nih.gov/articles/PMC4205553/).",
                styles,
            )
        )
        summary = computed.groupby("bin_ms").agg(
            recordings=("recording_name", "count"),
            total_edges=("correlation_edges_abs_gt_0_2", "sum"),
            mean_edges=("correlation_edges_abs_gt_0_2", "mean"),
            median_edges=("correlation_edges_abs_gt_0_2", "median"),
            mean_abs_correlation=("mean_abs_correlation", "mean"),
        ).reset_index()
        story.append(table_from_df(summary, styles))
        by_chip = computed.groupby(["bin_ms", "chip"]).agg(
            recordings=("recording_name", "count"),
            mean_edges=("correlation_edges_abs_gt_0_2", "mean"),
            median_edges=("correlation_edges_abs_gt_0_2", "median"),
            mean_abs_correlation=("mean_abs_correlation", "mean"),
        ).reset_index()
        story.append(table_from_df(by_chip, styles, max_rows=20))
        add_image(story, fig_dir / "correlation_bin_sensitivity.png", styles)

    story.append(paragraph("Per-Recording Metrics", styles, "Section"))
    story.append(table_from_df(metrics[["recording_name", "div", "chip", "ActiveElectrodesNumber", "NSpikesWell", "SpikesWellRate", "NBurstsWell", "BurstsSpikesPercentageWell"]].sort_values(["div", "chip"]), styles, max_rows=40))
    story.append(Spacer(1, 0.18 * cm))
    story.append(paragraph("Per-Recording Connectivity", styles, "Section"))
    story.append(table_from_df(conn[["recording_name", "div", "chip", "active_channels", "active_negative_spikes", "sttc_edges", "correlation_edges"]].sort_values(["div", "chip"]), styles, max_rows=40))

    doc = SimpleDocTemplate(str(out), pagesize=landscape(A4), leftMargin=1.0 * cm, rightMargin=1.0 * cm, topMargin=0.9 * cm, bottomMargin=0.9 * cm)
    doc.build(story)
    delivery_dir = Path(__file__).resolve().parents[1] / "output" / "pdf"
    delivery_dir.mkdir(parents=True, exist_ok=True)
    delivery_path = delivery_dir / f"experiment_{experiment_id}_very_detailed_with_methods_report.pdf"
    shutil.copy2(out, delivery_path)
    return out


def main():
    outputs = []
    build_3brain_figures()
    for experiment_id in (4, 8):
        fig_dir = build_mark_figures(experiment_id)
        outputs.append(pdf_appendix(experiment_id, fig_dir))
        outputs.append(build_mark_full_report(experiment_id, fig_dir))
        very_fig_dir = build_mark_very_detailed_figures(experiment_id)
        outputs.append(build_mark_very_detailed_report(experiment_id, very_fig_dir))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
