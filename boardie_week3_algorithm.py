from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "boston_boardie_analytics.csv"
RANDOM_STATE = 42


def create_demo_boardie_dataset(path: Path, n_rows: int = 1000) -> pd.DataFrame:
    """Creates a small reproducible demo dataset if the processed Week 2 file is not present.
    Replace this with the real Week 2 processed CSV when uploading to GitHub.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    room_types = np.array(["Entire home/apt", "Private room", "Shared room", "Hotel room"])
    neighbourhoods = np.array([
        "Allston", "Back Bay", "Beacon Hill", "Brighton", "Cambridge", "Dorchester",
        "Fenway", "Jamaica Plain", "Roxbury", "South Boston", "Mission Hill", "Somerville"
    ])
    property_types = np.array(["Apartment", "Condo", "House", "Dormitory-style room", "Studio"])

    room_type = rng.choice(room_types, n_rows, p=[0.45, 0.38, 0.10, 0.07])
    neighbourhood = rng.choice(neighbourhoods, n_rows)
    property_type = rng.choice(property_types, n_rows, p=[0.45, 0.18, 0.16, 0.12, 0.09])
    accommodates = rng.integers(1, 7, n_rows)
    bedrooms = np.maximum(1, np.round(accommodates / rng.uniform(1.7, 2.8, n_rows))).astype(int)
    beds = np.maximum(1, bedrooms + rng.integers(0, 3, n_rows))
    amenities_count = rng.integers(5, 42, n_rows)
    availability_365 = rng.integers(0, 366, n_rows)
    number_of_reviews = rng.poisson(22, n_rows)
    review_scores_rating = np.clip(rng.normal(4.55, 0.35, n_rows), 2.5, 5.0)
    availability_rate = availability_365 / 365
    host_is_superhost = rng.choice(["t", "f"], n_rows, p=[0.28, 0.72])
    instant_bookable = rng.choice(["t", "f"], n_rows, p=[0.34, 0.66])

    base_price = 55 + accommodates * 23 + bedrooms * 18 + amenities_count * 1.8
    premium_area = np.isin(neighbourhood, ["Back Bay", "Beacon Hill", "Fenway", "Cambridge"])
    budget_area = np.isin(neighbourhood, ["Dorchester", "Roxbury", "Allston"])
    base_price += premium_area * 55
    base_price -= budget_area * 25
    base_price += (room_type == "Entire home/apt") * 45
    base_price -= (room_type == "Shared room") * 45
    base_price += rng.normal(0, 30, n_rows)
    price = np.clip(base_price, 35, 700).round(2)

    df = pd.DataFrame({
        "id": np.arange(1, n_rows + 1),
        "host_is_superhost": host_is_superhost,
        "neighbourhood_cleansed": neighbourhood,
        "property_type": property_type,
        "room_type": room_type,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "beds": beds,
        "amenities_count": amenities_count,
        "availability_365": availability_365,
        "number_of_reviews": number_of_reviews,
        "review_scores_rating": review_scores_rating.round(2),
        "instant_bookable": instant_bookable,
        "availability_rate": availability_rate.round(3),
        "price": price,
    })
    df["price_band"] = pd.cut(
        df["price"],
        bins=[0, 100, 200, 350, 10000],
        labels=["Budget", "Mid-range", "Upper-mid", "Premium"],
        include_lowest=True,
    )
    df.to_csv(path, index=False)
    return df


if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded processed Week 2 dataset: {DATA_PATH}")
else:
    df = create_demo_boardie_dataset(DATA_PATH)
    print(f"Created demo processed dataset: {DATA_PATH}")

print("Dataset shape:", df.shape)

# The target label comes from Week 2 preprocessing.
# It turns numerical price into student-friendly categories for classification.
target_col = "price_band"
required_cols = [
    "host_is_superhost", "neighbourhood_cleansed", "property_type", "room_type",
    "accommodates", "bedrooms", "beds", "amenities_count", "availability_365",
    "number_of_reviews", "review_scores_rating", "instant_bookable", "availability_rate", target_col
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

model_df = df[required_cols].copy()
model_df = model_df.dropna(subset=[target_col])
model_df[target_col] = model_df[target_col].astype(str)

X = model_df.drop(columns=[target_col])
y = model_df[target_col]

numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# Baseline model: Decision Tree
baseline_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6))
])
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)

# Main model: Random Forest Classifier
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1))
])

param_grid = {
    "classifier__n_estimators": [50],
    "classifier__max_depth": [6, 10],
    "classifier__min_samples_leaf": [1, 2],
}

grid_search = GridSearchCV(
    rf_model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=1,
    verbose=0
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
rf_pred = best_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
report = classification_report(y_test, rf_pred)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, rf_pred, labels=labels)

results_text = f"""Boardie Week 3 Algorithm Results

Algorithm type: Classification
Target label: {target_col}
Baseline model: Decision Tree Classifier
Main model: Random Forest Classifier

Dataset rows used for modeling: {len(model_df):,}
Training rows: {len(X_train):,}
Testing rows: {len(X_test):,}

Baseline Decision Tree accuracy: {baseline_accuracy:.4f}
Fine-tuned Random Forest accuracy: {rf_accuracy:.4f}

Best Random Forest parameters:
{grid_search.best_params_}

Classification Report:
{report}
"""

(OUTPUT_DIR / "model_results.txt").write_text(results_text, encoding="utf-8")
print(results_text)

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(cm)
ax.set_title("Random Forest Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

# Feature importance plot
classifier = best_model.named_steps["classifier"]
pre = best_model.named_steps["preprocessor"]
feature_names = []
feature_names.extend(numeric_features)
if categorical_features:
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    feature_names.extend(ohe.get_feature_names_out(categorical_features).tolist())
importances = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(9, 6))
importances.sort_values().plot(kind="barh", ax=ax)
ax.set_title("Top 15 Feature Importances")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
plt.close(fig)

print("Saved outputs to:", OUTPUT_DIR)
