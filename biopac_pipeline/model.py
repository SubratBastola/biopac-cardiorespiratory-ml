
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .preprocess import get_preprocessing_pipeline

def build_model_pipeline():
    stacked_model = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ],
        final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced')
    )

    steps = get_preprocessing_pipeline(k=12)
    steps.append(("clf", stacked_model))
    return Pipeline(steps)
