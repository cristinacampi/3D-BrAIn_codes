Reproducibility
===============

Before Analysis
---------------

- Verify sampling rate, duration, grid convention, and reference channel.
- Assign repeated measurements to a stable biological-unit identifier.
- Keep separate experimental campaigns separate unless pooling is planned.
- Preserve raw-data and labels-file checksums.

During Analysis
---------------

- Save the exact command, random seed, and code revision.
- Report counts through each waveform and active-channel filter.
- Fit learned transformations only within training folds.
- Compare supervised models against their dummy baseline.
- Record connectivity windows, thresholds, sign rule, and displayed-edge rule.

Before Publication
------------------

- Review missing recordings and excluded classes.
- Report recording, biological-unit, class, DIV, channel, and spike counts.
- Distinguish cross-validation from an independent external test.
- Run sensitivity analyses for connectivity parameters.
- Run tests and build documentation with warnings treated as errors.
- Cite the software release and methodological publications.

The complete checklist is maintained in ``docs/REPRODUCIBILITY.md``.
