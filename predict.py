import joblib
import pandas as pd

# File Paths
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# Load Model & Pipeline
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

# Load Data for Prediction
input_data = pd.read_csv("test.csv")

# Keep original data
prediction_data = input_data.copy()

# Remove target column if present
if "median_house_value" in input_data.columns:
    input_features = input_data.drop(
        "median_house_value",
        axis=1
    )
else:
    input_features = input_data

# Transform Data
transformed_input = pipeline.transform(
    input_features
)

# Predict
predictions = model.predict(
    transformed_input
)

prediction_data["Predicted_House_Price"] = predictions

# Save Output
prediction_data.to_csv(
    "output.csv",
    index=False
)

print("Prediction completed successfully.")
print("Results saved to output.csv")