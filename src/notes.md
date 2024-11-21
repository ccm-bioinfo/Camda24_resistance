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
