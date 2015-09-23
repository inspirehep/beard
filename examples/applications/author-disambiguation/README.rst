This example shows how to build a full author disambiguation pipeline.
The pipeline is made of several scripts:

- ``sampling.py``: Build a training set of labeled pairs from a set of
  signatures, to be further used as input for ``distance.py``.::

    python sampling.py \
        --input_signatures input/signatures.json \
        --input_clusters input/clusters.json \
        --balanced 1 \
        --sample_size 1000000 \
        --output_pairs pairs/1M_nysiis_balanced.json \
        --use_blocking 1 \
        --blocking_function block_phonetic \
        --blocking_threshold 1 \
        --blocking_phonetic_alg nysiis \
        --verbose 1

- ``distance.py``: for inferring with supervised learning a distance or
  linkage function between signatures. An estimator is learned from
  labeled paired data and models whether two signatures belong to the same
  person.::

    python distance.py \
        --distance_pairs 1M_nysiis_balanced.json \
        --distance_model linkage.dat \
        --input_signatures input/signatures.json \
        --input_records input/records.json \
        --input_ethnicity_estimator ethnicity_estimator.pickle \
        --verbose 3

- ``clustering.py``: Semi-supervised block clustering, for grouping together
  signatures from the same author. Signatures are blocked and then clustered
  using hierarchical clustering together with the linkage function learned at
  the  previous step. For each block, the best cut-off threshold is chosen so
  as to maximize some scoring metric on the provided labeled data.::

    python clustering.py \
     --distance_model linkage.dat \
     --input_signatures input/signatures.json \
     --input_records input/records.json \
     --output_clusters predicted_clusters.json \
     --blocking_function block_phonetic \
     --blocking_threshold 0 \
     --blocking_phonetic_alg nysiis \
     --clustering_threshold 0.709 \
     --verbose 3 \
     --n_jobs 16

  If partial clusters are known, these should be specified using the
  ``input_clusters`` option.
