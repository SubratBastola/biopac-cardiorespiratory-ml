import pandas as pd
from sklearn.model_selection import StratifiedKFold

from biopac_pipeline.data_loader import load_data
from biopac_pipeline.preprocess import encode_labels, filter_variance
from biopac_pipeline.model import build_model_pipeline
from biopac_pipeline.evaluate import evaluate_pipeline

# === Load and preprocess ===
X, y = load_data("data/class_data_2022_2023_2024.csv")
y_encoded, le = encode_labels(y)
X = filter_variance(X, threshold=0.01)
class_names = le.classes_

# === Build and evaluate ===
pipeline = build_model_pipeline()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_auc_scores = evaluate_pipeline(pipeline, X, y_encoded, class_names, cv)

# === Final Summary ===
print(f"\n🏆 Mean ROC-AUC (multi-class OVR): {sum(roc_auc_scores)/len(roc_auc_scores):.2f}")

