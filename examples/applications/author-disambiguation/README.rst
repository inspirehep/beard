This example shows how to build a full author disambiguation pipeline.
The pipeline is made of two steps:

- Supervised learning, for inferring a distance or affinity function
  between publications. This estimator is learned from labeled paired data
  and models whether two publications have been authored by the same
  person. To perform supervised learning, run ``distance.py`` file.

- Semi-supervised block clustering, for grouping together publications
  from the same author. Publications are blocked by last name + first
  initial, and then clustered using hierarchical clustering together with
  the affinity function learned at the previous step. For each block,
  the best cut-off threshold is chosen so as to maximize some scoring
  metric on the provided labeled data. To perform clustering, run
  ``clustering.py`` file.
