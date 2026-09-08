from pathlib import Path
import math
import re

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("outputs/analisi_dati_3BRAIN")
STRAT_DIR = ROOT / "spike_stratification_negative_waveforms" / "26-06"
CLASS_DIR = ROOT / "spike_classification_negative_waveforms" / "26-06_recording_level"
CONN_DIR = ROOT / "connectivity_negative_spikes" / "26-06"
INTEGRATED_DIR = ROOT / "dataset_26-06_negative_integrated_report"

PAGE_SIZE = landscape(A4)
WIDTH, HEIGHT = PAGE_SIZE


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "TitleCenter",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=12,
    )
)
styles.add(ParagraphStyle("Section", parent=styles["Heading1"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle("Subsection", parent=styles["Heading2"], fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle("BodySmall", parent=styles["BodyText"], fontSize=8.4, leading=10.5, spaceAfter=4))
styles.add(ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=6.6, leading=7.8))
styles.add(ParagraphStyle("CellSmall", parent=styles["BodyText"], fontSize=5.6, leading=6.6))


def p(text, style="BodySmall"):
    return Paragraph(str(text), styles[style])


def fmt(value, digits=3):
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int,)) or (isinstance(value, float) and value.is_integer() and abs(value) < 1e9):
        return f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def clean_label_from_recording(name):
    text = str(name).lower()
    if "100um4ap_50umap5" in text or "50umap5_100um4ap" in text:
        return "excitation_inhibition"
    if "100um4ap" in text:
        return "excitation"
    if "50umap5" in text:
        return "inhibition"
    return "spontaneous"


def table_from_df(df, columns=None, max_rows=None, small=False):
    if columns is None:
        columns = list(df.columns)
    d = df.loc[:, columns].copy()
    if max_rows is not None:
        d = d.head(max_rows)
    cell_style = styles["CellSmall"] if small else styles["Cell"]
    data = [[p(c.replace("_", " "), "Cell") for c in columns]]
    for _, row in d.iterrows():
        data.append([Paragraph(fmt(row[c]), cell_style) for c in columns])
    available = WIDTH - 2.2 * cm
    col_width = available / max(1, len(columns))
    tbl = Table(data, colWidths=[col_width] * len(columns), repeatRows=1)
    tbl.setStyle(
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
    return tbl


def describe_numeric(df, columns):
    rows = []
    for col in columns:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s):
                rows.append(
                    {
                        "metric": col,
                        "sum": s.sum(),
                        "mean": s.mean(),
                        "median": s.median(),
                        "std": s.std(),
                        "min": s.min(),
                        "max": s.max(),
                    }
                )
    return pd.DataFrame(rows)


def add_title(story, title, subtitle):
    story.append(p(title, "TitleCenter"))
    story.append(p(subtitle, "BodySmall"))
    story.append(Spacer(1, 0.18 * cm))


def add_image(story, path, caption, width_cm=13.5):
    path = Path(path)
    if not path.exists():
        return
    img = Image(str(path))
    max_w = width_cm * cm
    max_h = 8.2 * cm
    scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    story.append(KeepTogether([img, p(caption, "Cell")]))
    story.append(Spacer(1, 0.15 * cm))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#546e7a"))
    canvas.drawString(1.1 * cm, 0.55 * cm, "3Brain 26-06 negative-spike analysis")
    canvas.drawRightString(WIDTH - 1.1 * cm, 0.55 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(path, story):
    doc = SimpleDocTemplate(
        str(path),
        pagesize=PAGE_SIZE,
        leftMargin=1.1 * cm,
        rightMargin=1.1 * cm,
        topMargin=0.9 * cm,
        bottomMargin=0.9 * cm,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def dataset_methods_story(title):
    story = []
    add_title(
        story,
        title,
        "Dataset 26-06, human iPSC brain organoids at 127 DIV. The reports summarize the uploaded outputs only: negative-spike stratification, negative-spike recording-level classification, and negative-spike connectivity.",
    )
    story.append(p("Dataset and Experimental Design", "Section"))
    story.append(
        p(
            "The 26-06 dataset contains 50 recordings from 17 organoids. Recording names encode the organoid identifier and pharmacological condition: spontaneous baseline, 100 uM 4AP excitation, 50 uM AP5 inhibition, and combined 4AP/AP5 or AP5/4AP perturbations. Each recording is treated as a 180 s MEA acquisition sampled at 19,753 Hz on a 64 x 64 electrode grid. Electrode 0 is excluded because it is the reference channel.",
        )
    )
    story.append(p("Negative-Spike Event Definition", "Section"))
    story.append(
        p(
            "All three downstream analyses use the same biologically plausible negative-spike universe. Candidate waveforms are first selected by detector label 0, then kept only when the negative trough dominates the positive peak by a factor of at least 1.10 and the trough lies within 10 samples of the waveform center. Active channels are channels with post-filter firing rate strictly greater than 0.25 Hz. This makes stratification, classification, and connectivity comparable because they are based on the same polarity and active-channel assumptions.",
        )
    )
    return story


def load_common():
    strat = pd.read_csv(STRAT_DIR / "per_recording_summary.csv")
    strat["label_normalized"] = strat["recording_name"].map(clean_label_from_recording)
    profile = pd.read_csv(STRAT_DIR / "recording_global_cluster_profile.csv")
    counts = pd.read_csv(STRAT_DIR / "recording_global_cluster_counts.csv")
    run_info = pd.read_csv(STRAT_DIR / "transversal_global_spike_clusters" / "run_info.csv")
    cls_sum = pd.read_csv(CLASS_DIR / "summary_metrics.csv")
    cv = pd.read_csv(CLASS_DIR / "cv_metrics.csv")
    conn = pd.read_csv(CONN_DIR / "connectivity_summary.csv")
    conn["label_normalized"] = conn["recording_name"].map(clean_label_from_recording)
    by_label = {
        "classification": pd.read_csv(INTEGRATED_DIR / "classification_data_by_label_summary.csv"),
        "connectivity": pd.read_csv(INTEGRATED_DIR / "connectivity_by_label_summary.csv"),
        "clusters": pd.read_csv(INTEGRATED_DIR / "global_cluster_profile_by_label.csv"),
        "stratification": pd.read_csv(INTEGRATED_DIR / "stratification_by_label_summary.csv"),
    }
    return strat, profile, counts, run_info, cls_sum, cv, conn, by_label


def stratification_report():
    strat, profile, counts, run_info, _, _, _, by_label = load_common()
    story = dataset_methods_story("Detailed Negative-Spike Stratification Report")
    story.append(p("Stratification Methodology", "Section"))
    story.append(
        p(
            "Waveforms are whitened and clustered using the Stratification module with PCA&KM. PCA components are retained until they explain 90 percent of waveform variance. K-means is run with Euclidean distance, represented in the code as Minkowski distance with p=2. Per-recording analyses are local to each recording, while the transversal/global analysis pools sampled plausible negative waveforms across recordings and reclusters them in a single PCA/K-means space. Candidate cluster counts are 2, 3, 4, 5, and 6; the uploaded global run selected 2 clusters.",
        )
    )
    story.append(
        p(
            "Sampling limits were MAX_SPIKES_PER_RECORDING=15,000 and MAX_SPIKES_GLOBAL=150,000 with RANDOM_STATE=42. Therefore, recordings with more than 15,000 active negative spikes contribute a subsample to clustering, while recordings below that threshold contribute all active negative spikes.",
        )
    )
    story.append(p("Run Configuration", "Subsection"))
    story.append(table_from_df(run_info.T.reset_index().rename(columns={"index": "parameter", 0: "value"}), ["parameter", "value"]))
    story.append(p("Dataset Coverage", "Section"))
    story.append(table_from_df(by_label["stratification"], small=True))
    story.append(p("Filter Retention by Condition", "Subsection"))
    retention = strat.groupby("label_normalized").agg(
        recordings=("recording_name", "count"),
        input_spikes=("n_input_spikes", "sum"),
        detector_negative=("n_after_label_filter", "sum"),
        plausible_negative=("n_after_waveform_filter", "sum"),
        active_negative=("n_after_active_filter", "sum"),
        sampled_spikes=("n_spikes", "sum"),
        mean_active_channels=("n_active_channels", "mean"),
    ).reset_index()
    retention["detector_negative_rate"] = retention["detector_negative"] / retention["input_spikes"]
    retention["plausible_given_detector_rate"] = retention["plausible_negative"] / retention["detector_negative"]
    retention["active_given_plausible_rate"] = retention["active_negative"] / retention["plausible_negative"]
    retention["sampled_given_active_rate"] = retention["sampled_spikes"] / retention["active_negative"]
    story.append(table_from_df(retention, small=True))
    story.append(p("Global Numeric Summary", "Subsection"))
    metrics = ["n_input_spikes", "n_after_label_filter", "n_after_waveform_filter", "n_after_active_filter", "n_spikes", "n_active_channels", "n_clusters"]
    story.append(table_from_df(describe_numeric(strat, metrics), small=True))
    story.append(p("Global Cluster Profile by Experimental Condition", "Section"))
    story.append(table_from_df(by_label["clusters"], small=True))
    add_image(story, INTEGRATED_DIR / "figures" / "stratification_global_cluster_profile_by_label.png", "Figure: fraction of global cluster assignments by experimental condition.")
    add_image(story, STRAT_DIR / "transversal_global_spike_clusters" / "pca_by_cluster.png", "Figure: global PCA projection of sampled plausible negative waveforms colored by transversal cluster.")
    add_image(story, STRAT_DIR / "transversal_global_spike_clusters" / "channel_maps" / "waveforms_number.png", "Figure: spatial distribution of waveforms contributing to the global clustering.")
    add_image(story, STRAT_DIR / "transversal_global_spike_clusters" / "channel_maps" / "cluster_0_waveform_percentage_by_channel.png", "Figure: percentage of local waveforms assigned to global cluster 0 by channel.")
    add_image(story, STRAT_DIR / "transversal_global_spike_clusters" / "channel_maps" / "cluster_1_waveform_percentage_by_channel.png", "Figure: percentage of local waveforms assigned to global cluster 1 by channel.")
    story.append(p("Per-Recording Results", "Section"))
    table_cols = ["recording_name", "label_normalized", "n_input_spikes", "n_after_waveform_filter", "n_after_active_filter", "n_spikes", "n_active_channels", "n_clusters"]
    story.append(table_from_df(strat.sort_values(["label_normalized", "recording_name"]), table_cols, max_rows=50, small=True))
    story.append(PageBreak())
    story.append(p("Sampling Cap Diagnostics", "Section"))
    cap = strat.copy()
    cap["hit_per_recording_cap"] = cap["n_after_active_filter"] > 15000
    cap["sampled_fraction_of_active"] = cap["n_spikes"] / cap["n_after_active_filter"]
    cap_summary = cap.groupby("label_normalized").agg(
        recordings=("recording_name", "count"),
        recordings_over_cap=("hit_per_recording_cap", "sum"),
        active_negative_spikes=("n_after_active_filter", "sum"),
        sampled_spikes=("n_spikes", "sum"),
        mean_sampled_fraction=("sampled_fraction_of_active", "mean"),
        min_sampled_fraction=("sampled_fraction_of_active", "min"),
    ).reset_index()
    story.append(table_from_df(cap_summary, small=True))
    story.append(
        p(
            "Recordings above 15,000 active plausible negative spikes are downsampled for clustering. This does not change the active-channel count, but it does reduce the number of waveforms used in PCA/K-means for high-activity recordings. The table below lists the recordings most affected by this cap.",
        )
    )
    story.append(table_from_df(cap.sort_values("sampled_fraction_of_active"), ["recording_name", "label_normalized", "n_after_active_filter", "n_spikes", "sampled_fraction_of_active", "n_active_channels"], max_rows=20, small=True))
    story.append(p("Per-Recording Global Cluster Counts", "Section"))
    merged_counts = counts.merge(strat[["recording_name", "label_normalized"]], on="recording_name", how="left")
    story.append(table_from_df(merged_counts.sort_values(["label_normalized", "recording_name"]), ["recording_name", "label_normalized", "0", "1"], max_rows=50, small=True))
    story.append(p("Per-Recording Global Cluster Fractions", "Section"))
    merged_profile = profile.merge(strat[["recording_name", "label_normalized"]], on="recording_name", how="left")
    story.append(table_from_df(merged_profile.sort_values(["label_normalized", "recording_name"]), ["recording_name", "label_normalized", "0", "1"], max_rows=50, small=True))
    story.append(p("Most Imbalanced Global Cluster Profiles", "Section"))
    imbalance = merged_profile.copy()
    imbalance["cluster_imbalance"] = (imbalance["0"] - imbalance["1"]).abs()
    story.append(table_from_df(imbalance.sort_values("cluster_imbalance", ascending=False), ["recording_name", "label_normalized", "0", "1", "cluster_imbalance"], max_rows=20, small=True))
    story.append(p("Interpretation Notes", "Section"))
    story.append(
        p(
            "The negative-polarity filter removes the previous trivial separation driven by positive versus negative waveforms. The remaining two-cluster global solution should be interpreted as waveform-shape stratification among plausible negative events. Because per-recording clusters are fitted independently, local cluster labels are not directly interchangeable across recordings; the transversal/global cluster profile is the appropriate object for cross-recording comparison.",
        )
    )
    out = STRAT_DIR / "negative_stratification_report_26-06_detailed_methods.pdf"
    build_pdf(out, story)
    return out


def classification_report():
    strat, _, _, _, cls_sum, cv, _, by_label = load_common()
    story = dataset_methods_story("Detailed Negative-Spike Recording-Level Classification Report")
    story.append(p("Classification Methodology", "Section"))
    story.append(
        p(
            "The classifier operates at recording level: each recording is one sample, and the target is the pharmacological condition inferred from the 26-06 label set. The split unit is the organoid, so recordings from the same organoid are held together during cross-validation. This design asks whether a recording-level feature vector can distinguish spontaneous, excitation, inhibition, and excitation-inhibition states while reducing leakage between conditions of the same organoid.",
        )
    )
    story.append(
        p(
            "Feature sets are activity-only, waveform-only, and combined. Activity features summarize active-channel firing and spike-count behavior. Waveform features include MiniRocket-derived time-series features from plausible negative waveforms, then are summarized at recording level. Combined features concatenate activity and waveform summaries. PCA retains 90 percent of variance before ridge or balanced logistic classification. The uploaded run used 10,000 MiniRocket kernels, 4 folds, 3 repeats, and 12 fold evaluations per model.",
        )
    )
    story.append(p("Input Data by Label", "Section"))
    story.append(table_from_df(by_label["classification"], small=True))
    add_image(story, INTEGRATED_DIR / "figures" / "active_negative_spikes_by_condition.png", "Figure: active plausible negative spikes by condition.")
    story.append(p("Cross-Validation Design", "Section"))
    cv_design = describe_numeric(cv, ["n_train_recordings", "n_test_recordings", "n_train_organoids", "n_test_organoids", "n_features", "n_pca_components", "num_kernels"])
    story.append(table_from_df(cv_design, small=True))
    story.append(p("Training/Test Size by Fold", "Subsection"))
    fold_sizes = cv.groupby(["repeat_index", "fold_index"]).agg(
        n_train_recordings=("n_train_recordings", "first"),
        n_test_recordings=("n_test_recordings", "first"),
        n_train_organoids=("n_train_organoids", "first"),
        n_test_organoids=("n_test_organoids", "first"),
    ).reset_index()
    story.append(table_from_df(fold_sizes, small=True))
    story.append(p("Model Ranking", "Section"))
    story.append(table_from_df(cls_sum.sort_values("mean_test_balanced_accuracy", ascending=False), small=True))
    add_image(story, INTEGRATED_DIR / "figures" / "classification_balanced_accuracy.png", "Figure: mean balanced accuracy by model and feature set.")
    story.append(p("Fold-Level Performance", "Section"))
    fold_summary = cv.groupby(["model", "feature_set", "classifier"]).agg(
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_min=("test_accuracy", "min"),
        test_accuracy_max=("test_accuracy", "max"),
        test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
        test_balanced_accuracy_min=("test_balanced_accuracy", "min"),
        test_balanced_accuracy_max=("test_balanced_accuracy", "max"),
        train_balanced_accuracy_mean=("train_balanced_accuracy", "mean"),
        n_folds=("fold_index", "count"),
    ).reset_index()
    story.append(table_from_df(fold_summary.sort_values("test_balanced_accuracy_mean", ascending=False), small=True))
    story.append(p("Complete Cross-Validation Metrics", "Section"))
    story.append(
        p(
            "The next table reports every fold evaluation. Repeated rows for each repeat/fold correspond to different feature-set/classifier combinations evaluated on the same organoid-level split."
        )
    )
    story.append(table_from_df(cv.sort_values(["repeat_index", "fold_index", "model"]), ["model", "repeat_index", "fold_index", "train_accuracy", "test_accuracy", "train_balanced_accuracy", "test_balanced_accuracy", "n_train_recordings", "n_test_recordings", "n_pca_components"], max_rows=72, small=True))
    story.append(PageBreak())
    story.append(p("Per-Class Reports and Confusion Matrices", "Section"))
    for _, row in cls_sum.sort_values("mean_test_balanced_accuracy", ascending=False).iterrows():
        model = row["model"]
        report_path = CLASS_DIR / f"classification_report_{model}.csv"
        conf_path = CLASS_DIR / f"confusion_matrix_{model}.csv"
        if report_path.exists():
            rep = pd.read_csv(report_path).rename(columns={"Unnamed: 0": "class"})
            story.append(p(model, "Subsection"))
            story.append(table_from_df(rep, small=True))
        if conf_path.exists():
            conf = pd.read_csv(conf_path, index_col=0).reset_index().rename(columns={"index": "true_label"})
            story.append(table_from_df(conf, small=True))
    story.append(p("Interpretation Notes", "Section"))
    best = cls_sum.sort_values("mean_test_balanced_accuracy", ascending=False).iloc[0]
    story.append(
        p(
            f"The best mean balanced accuracy is {best['mean_test_balanced_accuracy']:.3f} for {best['model']}. Performance is above a random four-class baseline but remains moderate, especially for inhibition and excitation classes. The gap between train and test balanced accuracy in logistic waveform/combined models suggests overfitting pressure from high-dimensional waveform features; ridge models generalize more stably in the uploaded results.",
        )
    )
    out = CLASS_DIR / "negative_recording_classification_report_26-06_detailed_methods.pdf"
    build_pdf(out, story)
    return out


def connectivity_report():
    _, _, _, _, _, _, conn, by_label = load_common()
    story = dataset_methods_story("Detailed Negative-Spike Connectivity Report")
    story.append(p("Connectivity Methodology", "Section"))
    story.append(
        p(
            "Connectivity is computed from spike times of active channels after the same plausible negative-spike filtering used for stratification and classification. Spike frames are converted to seconds using the 19,753 Hz sampling rate. Spike-count correlation uses 50 ms bins across the 180 s recording. STTC is computed with a 20 ms temporal window. Edges are retained when absolute spike-count correlation exceeds 0.2 or when STTC exceeds 0.2, depending on the network definition.",
        )
    )
    story.append(
        p(
            "For each recording, the pipeline saves the spike-count correlation matrix, STTC matrix, edge tables, node-strength tables, active-channel firing rates, binned spike counts, bin edges, and the per-channel negative-spike filter summary. The report summarizes these outputs at recording level and condition level.",
        )
    )
    story.append(p("Connectivity Coverage by Label", "Section"))
    story.append(table_from_df(by_label["connectivity"], small=True))
    story.append(p("Condition-Level Connectivity Distribution", "Subsection"))
    conn_distribution = conn.groupby("label_normalized").agg(
        recordings=("recording_name", "count"),
        active_channels_min=("active_channels", "min"),
        active_channels_median=("active_channels", "median"),
        active_channels_max=("active_channels", "max"),
        active_negative_spikes_min=("active_negative_spikes", "min"),
        active_negative_spikes_median=("active_negative_spikes", "median"),
        active_negative_spikes_max=("active_negative_spikes", "max"),
        sttc_edges_median=("sttc_edges", "median"),
        correlation_edges_median=("correlation_edges", "median"),
    ).reset_index()
    story.append(table_from_df(conn_distribution, small=True))
    add_image(story, INTEGRATED_DIR / "figures" / "connectivity_edges_by_label.png", "Figure: mean STTC and correlation edges by condition.")
    add_image(story, INTEGRATED_DIR / "figures" / "connectivity_spikes_vs_sttc_edges.png", "Figure: relationship between active negative spike count and STTC edge count.")
    story.append(p("Global Numeric Summary", "Section"))
    story.append(table_from_df(describe_numeric(conn, ["active_channels", "active_negative_spikes", "sttc_edges", "correlation_edges", "elapsed_seconds"]), small=True))
    story.append(p("Per-Recording Connectivity Results", "Section"))
    story.append(table_from_df(conn.sort_values(["label_normalized", "recording_name"]), ["recording_name", "label_normalized", "active_channels", "active_negative_spikes", "sttc_edges", "correlation_edges", "elapsed_seconds"], max_rows=50, small=True))
    story.append(PageBreak())
    story.append(p("Top Recordings by Network Density Proxy", "Section"))
    conn = conn.copy()
    conn["sttc_edges_per_active_channel"] = conn["sttc_edges"] / conn["active_channels"].replace(0, math.nan)
    conn["corr_edges_per_active_channel"] = conn["correlation_edges"] / conn["active_channels"].replace(0, math.nan)
    top_cols = ["recording_name", "label_normalized", "active_channels", "active_negative_spikes", "sttc_edges", "correlation_edges", "sttc_edges_per_active_channel", "corr_edges_per_active_channel"]
    story.append(table_from_df(conn.sort_values("sttc_edges", ascending=False), top_cols, max_rows=20, small=True))
    story.append(p("Sparse or Disconnected Recordings", "Section"))
    sparse = conn.sort_values(["sttc_edges", "correlation_edges", "active_negative_spikes"])
    story.append(table_from_df(sparse, top_cols, max_rows=20, small=True))
    story.append(p("Edge Agreement Between STTC and Correlation", "Section"))
    edge_agreement = conn.copy()
    edge_agreement["edge_count_difference_sttc_minus_corr"] = edge_agreement["sttc_edges"] - edge_agreement["correlation_edges"]
    edge_agreement["absolute_edge_count_difference"] = edge_agreement["edge_count_difference_sttc_minus_corr"].abs()
    story.append(table_from_df(edge_agreement.sort_values("absolute_edge_count_difference", ascending=False), ["recording_name", "label_normalized", "sttc_edges", "correlation_edges", "edge_count_difference_sttc_minus_corr", "absolute_edge_count_difference"], max_rows=25, small=True))
    story.append(p("Interpretation Notes", "Section"))
    story.append(
        p(
            "Excitation-inhibition and excitation recordings show the largest mean active-channel counts and highest active negative spike totals. Connectivity edge counts broadly follow this activity load; therefore edge counts should be interpreted together with active-channel and spike-count coverage, not as a pure connectivity change independent of firing rate.",
        )
    )
    out = CONN_DIR / "negative_connectivity_report_26-06_detailed_methods.pdf"
    build_pdf(out, story)
    return out


def integrated_report():
    strat, profile, counts, run_info, cls_sum, cv, conn, by_label = load_common()
    story = dataset_methods_story("Integrated Detailed Report: Dataset 26-06 Negative-Spike Analyses")
    story.append(p("Executive Summary", "Section"))
    total_input = strat["n_input_spikes"].sum()
    total_plausible = strat["n_after_waveform_filter"].sum()
    total_active = strat["n_after_active_filter"].sum()
    total_sampled = strat["n_spikes"].sum()
    story.append(
        p(
            f"The uploaded 26-06 analysis covers {len(strat)} recordings from 17 organoids at 127 DIV. Across recordings, {total_input:,} raw detected spikes were considered, {total_plausible:,} passed the detector-label and waveform-plausibility negative-spike filters, {total_active:,} remained on active channels, and {total_sampled:,} waveforms entered per-recording stratification after sampling caps. The global transversal clustering used {int(run_info.loc[0, 'n_spikes']):,} sampled waveforms and selected {int(run_info.loc[0, 'n_clusters'])} clusters.",
        )
    )
    story.append(p("Condition-Level Dataset Balance", "Section"))
    story.append(table_from_df(by_label["classification"], small=True))
    story.append(p("Methodological Consistency Across Analyses", "Section"))
    consistency = pd.DataFrame(
        [
            {"analysis": "stratification", "event_scope": "plausible negative spikes", "sample_unit": "waveform and recording", "active_channel_threshold_hz": 0.25, "reference_channel_excluded": True},
            {"analysis": "classification", "event_scope": "plausible negative spikes", "sample_unit": "recording", "active_channel_threshold_hz": 0.25, "reference_channel_excluded": True},
            {"analysis": "connectivity", "event_scope": "plausible negative spikes", "sample_unit": "channel network per recording", "active_channel_threshold_hz": 0.25, "reference_channel_excluded": True},
        ]
    )
    story.append(table_from_df(consistency, small=True))
    story.append(p("Stratification Results", "Section"))
    story.append(table_from_df(by_label["stratification"], small=True))
    story.append(table_from_df(by_label["clusters"], small=True))
    add_image(story, INTEGRATED_DIR / "figures" / "stratification_global_cluster_profile_by_label.png", "Figure: global cluster fractions by condition.")
    add_image(story, STRAT_DIR / "transversal_global_spike_clusters" / "pca_by_cluster.png", "Figure: global PCA projection of plausible negative waveform clusters.")
    story.append(p("Classification Results", "Section"))
    story.append(table_from_df(cls_sum.sort_values("mean_test_balanced_accuracy", ascending=False), small=True))
    add_image(story, INTEGRATED_DIR / "figures" / "classification_balanced_accuracy.png", "Figure: recording-level classification balanced accuracy.")
    story.append(p("Connectivity Results", "Section"))
    story.append(table_from_df(by_label["connectivity"], small=True))
    add_image(story, INTEGRATED_DIR / "figures" / "connectivity_edges_by_label.png", "Figure: connectivity edge counts by condition.")
    story.append(PageBreak())
    story.append(p("Cross-Analysis Recording-Level Consistency", "Section"))
    merged = strat.merge(conn, on="recording_name", suffixes=("_strat", "_conn"))
    merged["active_channel_difference"] = merged["n_active_channels"] - merged["active_channels"]
    merged["active_spike_difference"] = merged["n_after_active_filter"] - merged["active_negative_spikes"]
    story.append(
        p(
            "The consistency table compares active-channel and active-negative-spike counts between stratification and connectivity. Zero differences indicate that both analyses used the same post-filter event universe for that recording.",
        )
    )
    story.append(table_from_df(merged[["recording_name", "label_normalized_strat", "n_active_channels", "active_channels", "active_channel_difference", "n_after_active_filter", "active_negative_spikes", "active_spike_difference"]], max_rows=50, small=True))
    story.append(p("Per-Recording Master Table", "Section"))
    prof = profile.merge(counts, on="recording_name", suffixes=("_fraction", "_count"))
    master = merged.merge(prof, on="recording_name", how="left")
    cols = ["recording_name", "label_normalized_strat", "n_input_spikes", "n_after_waveform_filter", "n_after_active_filter", "n_spikes", "n_active_channels", "0_fraction", "1_fraction", "sttc_edges", "correlation_edges"]
    story.append(table_from_df(master.sort_values(["label_normalized_strat", "recording_name"]), cols, max_rows=50, small=True))
    story.append(PageBreak())
    story.append(p("Analysis-Specific Appendices", "Section"))
    story.append(p("Stratification Retention Appendix", "Subsection"))
    retention = strat.groupby("label_normalized").agg(
        recordings=("recording_name", "count"),
        input_spikes=("n_input_spikes", "sum"),
        detector_negative=("n_after_label_filter", "sum"),
        plausible_negative=("n_after_waveform_filter", "sum"),
        active_negative=("n_after_active_filter", "sum"),
        sampled_spikes=("n_spikes", "sum"),
    ).reset_index()
    retention["active_given_plausible_rate"] = retention["active_negative"] / retention["plausible_negative"]
    retention["sampled_given_active_rate"] = retention["sampled_spikes"] / retention["active_negative"]
    story.append(table_from_df(retention, small=True))
    story.append(p("Classification Fold Appendix", "Subsection"))
    fold_sizes = cv.groupby(["repeat_index", "fold_index"]).agg(
        n_train_recordings=("n_train_recordings", "first"),
        n_test_recordings=("n_test_recordings", "first"),
        n_train_organoids=("n_train_organoids", "first"),
        n_test_organoids=("n_test_organoids", "first"),
    ).reset_index()
    story.append(table_from_df(fold_sizes, small=True))
    story.append(p("Connectivity Density Appendix", "Subsection"))
    conn_density = conn.copy()
    conn_density["sttc_edges_per_active_channel"] = conn_density["sttc_edges"] / conn_density["active_channels"].replace(0, math.nan)
    conn_density["corr_edges_per_active_channel"] = conn_density["correlation_edges"] / conn_density["active_channels"].replace(0, math.nan)
    story.append(table_from_df(conn_density.sort_values("sttc_edges", ascending=False), ["recording_name", "label_normalized", "active_channels", "active_negative_spikes", "sttc_edges", "correlation_edges", "sttc_edges_per_active_channel", "corr_edges_per_active_channel"], max_rows=30, small=True))
    story.append(p("Overall Interpretation", "Section"))
    best = cls_sum.sort_values("mean_test_balanced_accuracy", ascending=False).iloc[0]
    story.append(
        p(
            f"The strongest condition-level separation is seen in activity and connectivity load: excitation-inhibition has the largest active negative spike count and mean active-channel count, while inhibition remains the sparsest condition. Global waveform clustering is less sharply condition-specific, with both clusters present in all labels. Recording-level classification reaches a best mean balanced accuracy of {best['mean_test_balanced_accuracy']:.3f}, indicating signal above chance but not a clean four-condition discriminator. Together, these outputs support a cautious interpretation: pharmacological perturbation changes the amount and spatial/network recruitment of plausible negative activity more clearly than it creates fully separable waveform classes.",
        )
    )
    out = INTEGRATED_DIR / "dataset_26-06_negative_integrated_report_detailed_methods.pdf"
    build_pdf(out, story)
    return out


def main():
    outputs = [
        stratification_report(),
        classification_report(),
        connectivity_report(),
        integrated_report(),
    ]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
