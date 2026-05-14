import json
import os
import pickle
import warnings
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE, SelectFromModel, SelectKBest, VarianceThreshold,
    f_classif, mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

warnings.filterwarnings('ignore')

RANDOM_STATE  = 42
TARGET_COLUMN = 'Attack_type'
np.random.seed(RANDOM_STATE)

os.makedirs('models',  exist_ok=True)
os.makedirs('results', exist_ok=True)

print("Libraries imported successfully!")
print(f"XGBoost available: {XGBOOST_AVAILABLE}")


# 2. Load Dataset

df = pd.read_csv('DNN-EdgeIIoT-dataset.csv', low_memory=False, on_bad_lines='skip')
print(f"Dataset shape: {df.shape}")


# Drop Features

COLUMNS_TO_DROP = [
    'frame.time', 'ip.src_host', 'ip.dst_host',
    'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
    'http.file_data', 'http.request.full_uri', 'icmp.transmit_timestamp',
    'http.request.uri.query', 'tcp.options', 'tcp.payload',
    'tcp.srcport', 'tcp.dstport', 'udp.port', 'mqtt.msg',
    'Attack_label',
]
for col in COLUMNS_TO_DROP:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"  Dropped '{col}'")

y = df[TARGET_COLUMN]
X = df.drop(TARGET_COLUMN, axis=1)

# Encode any remaining categorical feature columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Handle invalid values
X = X.replace([np.inf, -np.inf], 0).fillna(0)

# Encode target
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    target_names_original = le_target.classes_
else:
    target_names_original = np.unique(y)
    le_target = None

print(f"\nOriginal attack types ({len(target_names_original)}):")
for i, name in enumerate(target_names_original):
    count = np.sum(y == i)
    print(f"  {i:2d}: {name} ({count:,} samples)")


# Remove Duplicate Rows

initial_rows = df.shape[0]
duplicates   = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicates:,}")

df = df.drop_duplicates()
print(f"Rows removed:         {initial_rows - df.shape[0]:,}")
print(f"Shape after dedup:    {df.shape}")
print(f"\nAll features ({df.shape[1]}):")
print(df.columns.tolist())




GROUP_NAMES = ['Normal', 'DDoS', 'Reconnaissance', 'Web Attack', 'Malware/Access']

ATTACK_TO_GROUP = {
    'Backdoor':              4,
    'DDoS_HTTP':             1,
    'DDoS_ICMP':             1,
    'DDoS_TCP':              1,
    'DDoS_UDP':              1,
    'Fingerprinting':        2,
    'MITM':                  4,
    'Normal':                0,
    'Password':              4,
    'Port_Scanning':         2,
    'Ransomware':            4,
    'SQL_injection':         3,
    'Uploading':             3,
    'Vulnerability_scanner': 2,
    'XSS':                   3,
}

# Build numeric → group mapping from the fitted LabelEncoder
if le_target is not None:
    GROUP_MAP = {}
    for idx, name in enumerate(le_target.classes_):
        if name in ATTACK_TO_GROUP:
            GROUP_MAP[idx] = ATTACK_TO_GROUP[name]
        else:
            print(f"  WARNING: Unknown attack type '{name}' at index {idx} — skipping")
else:
    #must do otherwise the XGBOOST Will fail
    GROUP_MAP = {
        0: 4, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 4, 7: 0,
        8: 4, 9: 2, 10: 4, 11: 3, 12: 3, 13: 2, 14: 3,
    }

y_grouped = np.array([GROUP_MAP[val] for val in y])

print("Attack grouping applied:")
print(f"  {len(target_names_original)} original classes → {len(GROUP_NAMES)} grouped classes\n")

if le_target is not None:
    print("Mapping detail:")
    for idx, name in enumerate(le_target.classes_):
        grp = GROUP_MAP[idx]
        print(f"  {name:25s} (orig {idx:2d}) → {grp} ({GROUP_NAMES[grp]})")

print("\nGrouped class distribution:")
for grp_id, grp_name in enumerate(GROUP_NAMES):
    count = np.sum(y_grouped == grp_id)
    print(f"  {grp_id} — {grp_name:<16s}: {count:>8,}")
print(f"  Total: {len(y_grouped):,}")

y           = y_grouped
target_names = np.array(GROUP_NAMES)


#not needed but did before finding out, not needed
print('\nCleaning features...')

inf_count = np.isinf(X.values).sum()
if inf_count > 0:
    X = X.replace([np.inf, -np.inf], 0)
    print(f'  Replaced {inf_count:,} inf values')

nan_count = X.isnull().sum().sum()
if nan_count > 0:
    X = X.fillna(0)
    print(f'  Filled {nan_count:,} NaN values')

print(f'  Inf remaining: {np.isinf(X.values).sum()}')
print(f'  NaN remaining: {X.isnull().sum().sum()}')
print('Features cleaned.')




TOTAL_TARGET      = 125_000
SAMPLES_PER_CLASS = TOTAL_TARGET // n_classes 
MIN_SAMPLES     = 500
n_classes       = len(set(y))

print("=" * 80)
print(f"BALANCING DATASET — target: {SAMPLES_PER_CLASS:,} samples × {n_classes} classes = {TOTAL_TARGET:,} total")
print("=" * 80)

class_counts = Counter(y)
print("\nClass distribution before balancing:")
for cls in sorted(class_counts):
    print(f"  {cls} — {GROUP_NAMES[cls]:<16s}: {class_counts[cls]:>8,}")
print(f"  Total: {len(y):,}")

# Remove any class with fewer than MIN_SAMPLES (safety net)
classes_to_keep   = [cls for cls, count in class_counts.items() if count >= MIN_SAMPLES]
classes_to_remove = [cls for cls, count in class_counts.items() if count < MIN_SAMPLES]

if classes_to_remove:
    for cls in classes_to_remove:
        print(f"  WARNING: Removing {GROUP_NAMES[cls]} ({class_counts[cls]} samples < {MIN_SAMPLES})")
    mask = np.isin(y, classes_to_keep)
    X = X[mask].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X[mask]
    y = y[mask] if isinstance(y, np.ndarray) else y[mask].values
else:
    print(f"\nAll classes have >= {MIN_SAMPLES} samples.")


class_counts_after = Counter(y)
under_strategy = {
    cls: SAMPLES_PER_CLASS
    for cls, count in class_counts_after.items()
    if count > SAMPLES_PER_CLASS
}

if under_strategy:
    print(f"\nUndersampling {len(under_strategy)} classes:")
    for cls in sorted(under_strategy):
        print(f"  {GROUP_NAMES[cls]}: {class_counts_after[cls]:,} → {SAMPLES_PER_CLASS:,}")

    rus  = RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_STATE)
    X, y = rus.fit_resample(X, y)

remaining_classes = sorted(np.unique(y))
label_map    = {old: new for new, old in enumerate(remaining_classes)}
target_names = np.array([GROUP_NAMES[cls] for cls in remaining_classes])
y = np.array([label_map[val] for val in y])

final_counts = Counter(y)
print(f"\n{'=' * 80}")
print("FINAL BALANCED DISTRIBUTION")
print("=" * 80)
for cls in sorted(final_counts):
    print(f"  {cls} — {target_names[cls]:<16s}: {final_counts[cls]:>8,}")
print(f"\n  Total:   {len(y):,}")
print(f"  Classes: {len(final_counts)}")
print("Dataset balanced.")

le_final = LabelEncoder()
y        = le_final.fit_transform(y)

print(f"\nFinal class mapping:")
for i, name in enumerate(target_names):
    print(f"  {i}: {name}")
print("Target re-encoded.")


#SCALING

feature_names = X.columns.tolist()
print(f"\nTotal features: {len(feature_names)}")

scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

print(f"MinMaxScaler applied — shape: {X_scaled.shape}")

# Final NaN guard
nan_in_X = X_scaled.isnull().sum().sum()
nan_in_y = pd.Series(y).isnull().sum()
print(f"NaN in features: {nan_in_X}")
print(f"NaN in target:   {nan_in_y}")

if nan_in_X > 0 or nan_in_y > 0:
    print("WARNING: NaN detected — filling with 0...")
    X_scaled = X_scaled.fillna(0)
    y        = pd.Series(y).fillna(0).values
    print("NaN values handled.")

assert X_scaled.isnull().sum().sum() == 0, "ERROR: NaN still in features!"
assert pd.Series(y).isnull().sum()   == 0, "ERROR: NaN still in target!"
print("No NaN values — ready for split.")


#TEST TRAIN

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=5000, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set:     {X_test.shape}")

print("\nClass distribution (train):")
for cls, count in sorted(Counter(y_train).items()):
    print(f"  {cls} — {target_names[cls]:<16s}: {count:,}")

print("\nClass distribution (test):")
for cls, count in sorted(Counter(y_test).items()):
    print(f"  {cls} — {target_names[cls]:<16s}: {count:,}")


#Save Splits + Metadata

os.makedirs('grouped_dataset', exist_ok=True)

X_train.to_csv('grouped_dataset/X_train.csv', index=False)
X_test.to_csv( 'grouped_dataset/X_test.csv',  index=False)
pd.DataFrame(y_train, columns=['Attack_type']).to_csv('grouped_dataset/y_train.csv', index=False)
pd.DataFrame(y_test,  columns=['Attack_type']).to_csv('grouped_dataset/y_test.csv',  index=False)

class_info = {
    'group_names':   GROUP_NAMES,
    'target_names':  target_names.tolist(),
    'num_classes':   len(target_names),
    'samples_train': int(len(y_train)),
    'samples_test':  int(len(y_test)),
    'features':      feature_names,
}


print("\nDatasets saved to grouped_dataset/")
print(f"  X_train.csv     : {X_train.shape}")
print(f"  X_test.csv      : {X_test.shape}")
print(f"  y_train.csv     : ({len(y_train)},)")
print(f"  y_test.csv      : ({len(y_test)},)")
print(f"  class_info.json")
