from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    "logistic_regression": OneVsRestClassifier(LogisticRegression()),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgboost": XGBClassifier(n_estimators=100, random_state=42),
    "lightgbm": OneVsRestClassifier(LGBMClassifier(n_estimators=100, random_state=42)),
    "catboost": MultiOutputClassifier(CatBoostClassifier(random_seed=42, verbose=0)),
    "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "gradient_boosting": OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100, random_state=42)),
    "knn": OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5)),  # Wrapped in OneVsRest
    "svm": OneVsRestClassifier(SVC(probability=True, random_state=42))  # Wrapped in OneVsRest
}
