import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# File Paths
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


# Build Preprocessing Pipeline

def build_pipeline(num_attribs, cat_attribs):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline


# Load Dataset
housing = pd.read_csv("data/housing.csv")


# Stratified Sampling
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_index, test_index in split.split(housing, housing["income_cat"]):

    train_set = housing.loc[train_index].drop("income_cat", axis=1)
    test_set = housing.loc[test_index].drop("income_cat", axis=1)

# Save test set for prediction
test_set.to_csv("test.csv", index=False)


# Separate Features & Labels
housing_labels = train_set["median_house_value"].copy()

housing_features = train_set.drop(
    "median_house_value",
    axis=1
)

# Numerical & Categorical Columns
num_attribs = housing_features.drop(
    "ocean_proximity",
    axis=1
).columns.tolist()

cat_attribs = ["ocean_proximity"]

# Data Preprocessing
pipeline = build_pipeline(
    num_attribs,
    cat_attribs
)

housing_prepared = pipeline.fit_transform(
    housing_features
)

# Model
model = RandomForestRegressor(
    random_state=42
)

# Cross Validation
scores = cross_val_score(
    model,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=5
)

rmse_scores = np.sqrt(-scores)

print("="*50)
print("Cross Validation RMSE")
print(rmse_scores)
print("Average RMSE:", rmse_scores.mean())
print("="*50)

# Train Final Model
model.fit(
    housing_prepared,
    housing_labels
)


# Evaluate on Test Set
X_test = test_set.drop(
    "median_house_value",
    axis=1
)

y_test = test_set["median_house_value"].copy()

X_test_prepared = pipeline.transform(X_test)

predictions = model.predict(X_test_prepared)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nTest RMSE :", rmse)
print("Test R² :", r2)


# Feature Importance
feature_names = pipeline.get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")
print(importance.head(10))

# Save Model & Pipeline
joblib.dump(model, MODEL_FILE)
joblib.dump(pipeline, PIPELINE_FILE)

print("\nModel Saved Successfully.")