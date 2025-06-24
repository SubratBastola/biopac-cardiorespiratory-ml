
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

def filter_variance(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = X.loc[:, selector.fit(X).get_support()]
    return X_filtered

def get_preprocessing_pipeline(k=12):
    return [
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k))
    ]
