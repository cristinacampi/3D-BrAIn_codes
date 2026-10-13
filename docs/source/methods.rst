Scientific Methods
==================

Plausible Negative Events
-------------------------

The shared filter retains detector label ``0`` when labels are available,
requires an absolute negative trough at least 1.10 times the absolute positive
peak, and requires the trough within 10 samples of the waveform centre. These
are operational quality-control rules, not a universal physiological
definition. Non-reference channels must have a post-filter firing rate strictly
greater than 0.25 Hz by default.

Bursts
------

A burst contains at least five consecutive spikes with each inter-spike
interval no greater than 0.1 seconds by default. The public parameters are
``--burst-min-spikes`` and ``--burst-max-isi``. Network burst calculation is
available through ``BrwFunctions.NetworkBurstMetric`` but is not part of the
metrics main.

Stratification
--------------

Waveforms are whitened, projected onto enough principal components to explain
90% of variance, and clustered by k-means with Euclidean distance. Candidate
cluster counts are configurable. At most 15,000 events per recording and
150,000 pooled events per experiment are sampled by default. Per-recording and
transversal analyses are both produced.

Classification and Regression
-----------------------------

One recording is one sample. Cross-validation is group-disjoint by
``biological_unit_id``. Classification is stratified by class; DIV regression
uses discretised DIV only to balance folds and predicts continuous DIV.
MiniRocket is fitted inside each training fold and transformed waveform features
are aggregated per recording by feature-wise mean and standard deviation.

Balanced accuracy is the unweighted mean recall across classes. Test metrics
are held-out cross-validation metrics unless an independent external dataset is
explicitly supplied and reported.

Connectivity
------------

Spike-count correlation uses 50 ms count bins by default. STTC uses a 20 ms
coincidence window. Edges pass an absolute weight threshold of 0.2 by default;
sensor maps display the strongest 5% of passing edges, capped at 120. All of
these values are configurable parameters and should be included in sensitivity
analysis.

For the complete methods narrative and references, consult ``docs/METHODS.md``
in the repository.
