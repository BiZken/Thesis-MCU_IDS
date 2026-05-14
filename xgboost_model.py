import json
import os
import pickle
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

warnings.filterwarnings('ignore')

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

os.makedirs('models',  exist_ok=True)
os.makedirs('results', exist_ok=True)

print("Libraries imported successfully!")
print(f"XGBoost available: {XGBOOST_AVAILABLE}")

X_train = pd.read_csv('X_train.csv').values
X_test  = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test  = pd.read_csv('y_test.csv').values.ravel()

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {np.unique(y_train)}")

models      = {}
param_grids = {}

models['XGBoost'] = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')
param_grids['XGBoost'] = {
    'n_estimators':     list(range(2, 21)),
    'max_depth':        [1, 2, 3, 4, 5],
    'learning_rate':    [0.01, 0.05, 0.1],
    'subsample':        [0.5, 0.6, 0.7],
    'colsample_bytree': [0.6, 0.7, 0.8],
}

print(f"Total models to train: {len(models)}")
for name in models:
    print(f"  - {name}")

results        = {}
trained_models = {}
best_params    = {}

print("Starting model training...\n")

for model_name, model in models.items():
    print("=" * 60)
    print(f"Training {model_name}...")
    print("=" * 60)

    grid_search = GridSearchCV(
        model,
        param_grids[model_name],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=3,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    trained_models[model_name] = best_model
    best_params[model_name]    = grid_search.best_params_

    inference_times = []
    for i in range(len(X_test)):
        sample = X_test[i:i+1]
        t0 = time.perf_counter()
        best_model.predict(sample)
        inference_times.append((time.perf_counter() - t0) * 1000)

    inference_times  = np.array(inference_times)
    mean_inference   = np.mean(inference_times)
    max_inference    = np.max(inference_times)
    min_inference    = np.min(inference_times)

    y_pred    = best_model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f2        = fbeta_score(y_test, y_pred, beta=2, average='macro', zero_division=0)

    results[model_name] = {
        'accuracy':               accuracy,
        'precision':              precision,
        'recall':                 recall,
        'f1_score':               f1,
        'f2_score':               f2,
        'best_params':            grid_search.best_params_,
        'best_cv_score':          grid_search.best_score_,
        'confusion_matrix':       confusion_matrix(y_test, y_pred).tolist(),
        'classification_report':  classification_report(y_test, y_pred, output_dict=True),
        'inference_time_mean_ms': mean_inference,
        'inference_time_max_ms':  max_inference,
        'inference_time_min_ms':  min_inference,
    }

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score:   {grid_search.best_score_:.4f}")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  F2-Score:  {f2:.4f}")
    print(f"\nInference Time (per sample):")
    print(f"  Mean: {mean_inference:.4f} ms")
    print(f"  Max:  {max_inference:.4f} ms")
    print(f"  Min:  {min_inference:.4f} ms\n")

results_df = pd.DataFrame({
    'Model':         list(results.keys()),
    'Accuracy':      [results[m]['accuracy']       for m in results],
    'Precision':     [results[m]['precision']      for m in results],
    'Recall':        [results[m]['recall']         for m in results],
    'F2-Score':      [results[m]['f2_score']       for m in results],
    'Best_CV_Score': [results[m]['best_cv_score']  for m in results],
})
results_df = results_df.sort_values('Accuracy', ascending=False)
for model_name, model in trained_models.items():
    path = f'models/{model_name}_model.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved: {path}")

print(f"\nAll {len(trained_models)} models saved!")

with open('results/xgboost_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("Saved: results/xgboost_results.json")

results_df.to_csv('results/xgboost_summary.csv', index=False)
print("Saved: results/xgboost_summary.csv")

with open('results/xgboost_best_parameters.json', 'w') as f:
    json.dump(best_params, f, indent=4)
print("Saved: results/xgboost_best_parameters.json")
