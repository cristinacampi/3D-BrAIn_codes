# Scientific Methods

This chapter describes the current implementation and is suitable as a basis
for a manuscript's computational methods section. Replace software version,
dataset counts, and configured parameters with the values used in the final
analysis. The software is intended for research use and does not itself
establish biological or clinical validity.

## Signal Extraction and Spike Detection

Raw 3Brain BRW recordings are exported to chunked HDF5 channel traces. Spike
detection applies notch and band-pass filtering, with public defaults of 50 Hz,
300 Hz, and 3000 Hz, respectively. Events are detected at a default threshold
factor of 5 and stored as odd-length waveform windows (41 samples by default),
detection frames, and polarity labels. The sampling rate is read from the raw
recording during extraction; downstream analyses must be given the matching
rate when it cannot be inferred.

## Plausible Negative Waveforms

Stratification, classification, regression, connectivity, and saved-model
application use `SpikeSorting.NegativeWaveformMask`. An event is retained when:

1. its detector label is `0`, when finite labels are available;
2. the absolute negative trough is at least 1.10 times the absolute positive
   peak; and
3. the trough occurs within 10 samples of the waveform centre.

The amplitude and centring rules remove events whose positive component
dominates or whose extremum is inconsistent with the extracted event centre.
They are operational quality-control criteria, not a universal physiological
definition. Their effect must be reported as counts before and after each
filter. When detector labels are absent, the waveform criteria still apply.

Channel `0` is excluded as the reference. A channel is active only when the
post-filter event rate is strictly greater than 0.25 Hz by default:

```text
firing rate = retained events / recording duration in seconds.
```

## Spike and Burst Metrics

Spike metrics are computed per channel and at well level. They include event
count and rate, inter-spike interval (ISI), ISI variability, peak-to-peak
amplitude, and active-electrode summaries.

A burst is a maximal sequence containing at least five spikes by default in
which consecutive ISIs are no greater than 0.1 s. Once the initial five-spike
condition is met, the burst is extended while the ISI criterion continues to
hold. Burst outputs include burst count and rate, duration, spikes per burst,
within-burst ISI, inter-burst interval (IBI), and the percentage of spikes in
bursts. Parameters are exposed as `--burst-min-spikes` and `--burst-max-isi`.

`BrwFunctions.NetworkBurstMetric` is available as an API function. It detects
time windows containing bursts on a specified fraction of active electrodes.
It is not called by `main_compute_spike_burst_metrics.py`; network-burst values
should therefore not be claimed unless that function is run explicitly and its
parameters are reported.

## Waveform Stratification

Waveforms are analysed separately for every recording and, within each
experiment identifier inferred from the spike-root path, in a transversal
pooled analysis. Distinct experimental campaigns should be passed separately
when they are not intended to share a clustering basis.

After shared filtering and active-channel selection, at most 15,000 waveforms
per recording are selected by seeded random sampling. The transversal analysis
uses at most 150,000 pooled waveforms per experiment. Waveforms are whitened and
passed to `Stratification.Clustering` with algorithm `PCA&KM`, Euclidean
distance (`m`, Minkowski p = 2), and candidate cluster counts 2 through 6 by
default. PCA retains the minimum number of components whose cumulative
explained variance reaches 90%. K-means is then evaluated over the requested
candidate counts according to the implementation in `Stratification.py`.

Outputs include spike assignments, centroid waveforms with shaded one-standard-
deviation bands, PCA visualisation, cluster-by-channel tables, waveform-count
maps, and the percentage of each channel's waveforms assigned to each cluster.
The global profile table gives, for every recording, the proportion assigned to
each transversal cluster.

The displayed two-dimensional PCA plot is a visual projection and does not
imply that clustering used only two components.

## Recording-Level Classification

The statistical sample is one recording, not one spike. Each recording is
represented by activity features, scalar waveform summaries, and optionally
MiniRocket waveform features. Activity features include retained spike counts,
firing rate, active-channel counts, channel count statistics, and spatial
entropy. Waveform summaries include global mean, standard deviation, minimum,
maximum, peak-to-peak range, and energy.

MiniRocket is fitted only on waveforms from the training fold. Up to 12,000
training waveforms are used to fit 10,000 kernels by default. Every waveform is
transformed into kernel responses; responses are then aggregated within each
recording by feature-wise mean and standard deviation. Thus 10,000 kernels
produce 20,000 recording-level MiniRocket variables before concatenation with
other selected features.

Classical estimators include a most-frequent baseline, ridge classifier,
class-balanced logistic regression, class-balanced linear and radial-basis SVM,
random forest, and Extra Trees. Linear and SVM pipelines standardise features
and use PCA retaining 90% explained variance. Tree ensembles use 300 trees by
default.

Mamba uses a five-channel sequence per recording: the pointwise waveform mean,
standard deviation, median, 25th percentile, and 75th percentile. The portable
reference implementation uses PyTorch; the optional official backend uses
`mamba-ssm` when available.

Validation uses repeated `StratifiedGroupKFold`. The default is four folds and
three repeats. The class label determines stratification, while
`biological_unit_id` determines groups. Consequently, recordings from one
organoid or other biological unit cannot occur in both training and test parts
of a fold. Feature extraction that learns from data, including MiniRocket and
PCA, is fitted within the training fold.

Balanced accuracy is the unweighted mean of recall across classes:

```text
balanced accuracy = (1 / K) * sum_k TP_k / (TP_k + FN_k).
```

This gives each of `K` classes equal weight regardless of prevalence. Accuracy,
confusion matrices, per-class precision/recall/F1, and train/test differences
should be interpreted alongside balanced accuracy. Reported test results are
cross-validated held-out-fold results and are not an independent external test.

## DIV Regression

DIV is treated as a continuous target. Feature construction and the
plausible-negative event universe match classification. Models include a mean
baseline, ridge, Elastic Net, linear and RBF support-vector regression, random
forest, and Extra Trees. PCA-based pipelines retain 90% explained variance.

Cross-validation is group-disjoint by biological unit. DIV values are binned
into quantiles only to improve fold balance; the estimators predict the original
continuous DIV. If stratified grouped splitting is infeasible, the pipeline
falls back to `GroupKFold`. Metrics include mean and median absolute error,
RMSE, R-squared, Pearson and Spearman association, and proportions within 5,
10, and 15 days.

## Functional Connectivity

Connectivity is computed from the retained spike times of active channels.
Spike-count correlation bins each train into fixed windows (50 ms by default)
and computes pairwise correlation between channel count vectors. This captures
co-fluctuation at the selected timescale and is therefore sensitive to bin
width, firing rate, non-stationarity, and common drive.

STTC is computed with a 20 ms coincidence window by default. It compares the
fraction of spikes close to the other train with the fraction of recording time
covered by coincidence windows, reducing the firing-rate bias of simpler
coincidence measures. Values range approximately from -1 to 1, with positive
values indicating temporal association, values near zero indicating little
association, and negative values indicating avoidance.

Edges pass the default threshold when their absolute weight is at least 0.2.
With `--signed-edges`, thresholding instead uses signed values. The map then
shows the strongest 5% of threshold-passing edges, capped at 120 edges by
default. "Strongest" means greatest absolute edge weight unless signed-edge
selection is requested. Global node strength is the sum of absolute weights in
the full matrix; selected-map node strength is calculated from displayed edges
only. A node may therefore be globally strong but have no displayed edge after
thresholding, percentile selection, or edge-count capping.

Correlation bin width, STTC coincidence window, thresholds, edge sign rule,
displayed percentage, and edge cap are all configurable parameters. A
sensitivity analysis should vary at least the correlation bin, STTC window, and
edge threshold, then compare edge density, node rankings, and spatial network
stability. Parameter exploration must be distinguished from confirmatory
analysis.

## Model Persistence and New Data

After cross-validation, selected models are refitted on all eligible labelled
recordings and saved under `final_models/`. Each bundle stores the estimator,
feature-extraction objects where applicable, label mapping, preprocessing
configuration, training-recording manifest, estimator parameters, and
cross-validation metrics. The final fit has no unseen internal test subset; its
reported generalisation estimate remains the prior grouped cross-validation.

Applying a saved classifier repeats negative-waveform selection and
recording-level feature extraction using the saved bundle. New data must match
the training sampling assumptions, duration interpretation, waveform length,
channel convention, and biological domain.

## Recommended Citations

- Dempster A, Schmidt DF, Webb GI. *MiniRocket: A Very Fast (Almost)
  Deterministic Transform for Time Series Classification*. KDD 2021.
  <https://arxiv.org/abs/2012.08791>
- Cutts CS, Eglen SJ. *Detecting pairwise correlations in spike trains: an
  objective comparison of methods and application to the study of retinal
  waves*. Journal of Neuroscience, 2014. <https://doi.org/10.1523/JNEUROSCI.2767-14.2014>
- Gu A, Dao T. *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces*. 2023. <https://arxiv.org/abs/2312.00752>
- Pedregosa F et al. *Scikit-learn: Machine Learning in Python*. Journal of
  Machine Learning Research, 2011.

Also cite the 3D-BrAIn software release using `CITATION.cff` and cite all
dataset, organoid, acquisition, and spike-detection methods used upstream.
