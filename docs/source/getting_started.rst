Getting Started
===============

Import the module that matches the workflow you are running:

.. code-block:: python

   from brain_3d import brw_functions as brw
   from brain_3d import bxr_functions as bxr
   from brain_3d import stratification

Read recording and analysis files:

.. code-block:: python

   recording = brw.ReadBRW("recording.BRW", "Well_A1")
   spikes = bxr.ReadBXR("analysis.BXR", "Well_A1")

Cluster extracted waveforms or feature matrices:

.. code-block:: python

   clusters = stratification.Clustering(
       Data,
       Algo="KM",
       DistanceStr="m",
       MaxClasses=[2, 3, 4],
   )

For fuzzy C-means directly:

.. code-block:: python

   import numpy as np
   from brain_3d import FCM
   from brain_3d.stratification import d_m

   data = np.asarray([[0.0, 0.1], [0.2, 0.0], [5.0, 5.1], [5.2, 4.9]])
   initial_centers = [data[0], data[2]]
   clusters, centers, membership = FCM.FCM(
       data,
       NClasses=2,
       Centers=initial_centers,
       FuzzyParameter=2,
       MaxIter=10,
       Metric=d_m,
   )
