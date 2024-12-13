create new branch & delete everything
set up conda environment.yml for sklearn, xgboost and dev dependencies and an environment in .conda

make file (env, install, formatting, linting, testing)
pre-commit

vscode settings
ruff

Dockerfile for python 3.10

devcontainer.json

data - download data
prepare data
preprocess data (drop, impute, encode, scale)

model the data

- baseline

  - dummy classifier
  - dummy regressor

- model 1

  - random forest regression
  - random forest classifier

- model 2

  - xgboost regression
  - xgboost classifier

- model 3

  - stage 1: classify dominant class
  - stage 2: regression on non-dominant class

- model 3 ensemble

  - stacking

- model 4

  - neural network

log models and run nested cross validation

# subset data
# filter, antibiotic-selector (first, second or both) <-- transformer
# treat imbalance
# imbalance, random oversampling  <-- transformer
# features
# feature, encoding <-- transformer
# targets
# target_mic_task_type <-- variable ("classification", "regression")
# target_mic, encoding, standardization <-- target transformer
# target_phenotype, encoding <-- target transformer
# sub-estimators
# mic_estimator, standardization/encoding <-- estimator depends on target_mic_type
# phenotype_estimator <-- estimator, classifier/regressor
# meta-estimator
# meta_estimator, subestimators instances from mic & phenotype <-- metaestimator, estimator 
# scores
#### score: custom function inside the meta estimator 
# chaining
#### possibly chain classifiers of mic and phenotype (both ways) incorporated in the meta estimator
