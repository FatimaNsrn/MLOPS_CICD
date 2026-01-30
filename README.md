# MLOps CI/CD Pipeline for Product Profitability Prediction

This repository contains an academic MLOps project focused on building and validating a machine-learning prediction service for product profitability and automating testing and deployment using GitHub Actions.

The business task is to predict whether a product–store combination will result in:

loss

low profit

high profit

based on pricing, promotions, seasonality, demand history, and store/product information.

## Project Overview

The project evolved in two stages:

Regression phase: predicting numeric profit margin

Logistic Regression

CatBoost

Classification phase: converting profit margin into buckets (loss / low / high)

CatBoost

Neural Network with embeddings

High-cardinality features such as product_id and store_id were handled using:

feature hashing

embeddings (for neural models)

All experiments and models were logged and compared using MLflow.

The final deployed model for this repository is the CatBoost classifier.

## Training

The training/ folder contains notebooks for:

Logistic Regression

CatBoost (regression & classification)

Neural Network classifier

Final model selection experiments

MLflow is used to store:

experiment runs

metrics

artifacts

registered models

## Feature Engineering

The features/ directory contains the hashing function used for high-cardinality categorical variables.

This function is reused during training and serving and is covered by unit tests to ensure deterministic behavior.

## Model Serving

The serving/ directory includes:

FastAPI applications (main.py, main2.py)

Docker configuration

schema definitions

test_predict.py client script

The API loads the selected model from MLflow and exposes endpoints for prediction.

## Testing Strategy

This project includes:

Unit Tests

validate the hashing function

Smoke Tests

start the containerized service

call prediction endpoints

verify correct responses

## CI/CD with GitHub Actions

GitHub Actions workflows in .github/workflows/ implement automated checks for every push or pull request.

They include:

dependency installation

unit tests

Docker build

service startup

smoke test execution

To verify that CI was working correctly, an intentional bug was introduced into the hashing function.
The pipeline failed as expected, blocking progress.
After fixing the bug, the pipeline passed again.
