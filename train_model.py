
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create folder for graphs
os.makedirs("static/graphs", exist_ok=True)

# 2. Load dataset
data = pd.read_csv("insurance.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Display available columns
print("Dataset columns:", data.columns.tolist())

# 3. Check required columns
required_columns = [
    "age", "sex", "bmi", "children",
    "smoker", "region", "expenses"
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}. "
        "Please check your insurance.csv file."
    )

# 4. Select input and target columns
X = data[["age", "sex", "bmi", "children", "smoker", "region"]]
y = data["expenses"]

# 5. Prepare categorical data
categorical = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("categories",
         OneHotEncoder(handle_unknown="ignore"),
         categorical)
    ],
    remainder="passthrough"
)

# 6. Create Linear Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", LinearRegression())
])

# 7. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train the model
model.fit(X_train, y_train)

# 9. Predict insurance expenses
predictions = model.predict(X_test)

# 10. Display model performance
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel trained successfully!")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 3))

# GRAPH 1: Actual vs Predicted Cost
plt.figure(figsize=(8, 5))

plt.scatter(y_test, predictions, alpha=0.6)

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Insurance Expenses")
plt.ylabel("Predicted Insurance Expenses")
plt.title("Actual vs Predicted Insurance Cost")
plt.tight_layout()

plt.savefig("static/graphs/actual_vs_predicted.png")
plt.close()

# GRAPH 2: Feature Coefficients
feature_names = (
    model.named_steps["preprocessor"].get_feature_names_out()
)

coefficients = model.named_steps["regression"].coef_

features = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Select the 10 features with the largest absolute coefficients
top_features = (
    features.assign(
        AbsoluteCoefficient=features["Coefficient"].abs()
    )
    .nlargest(10, "AbsoluteCoefficient")
    .sort_values("Coefficient")
)

plt.figure(figsize=(9, 6))

plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

plt.xlabel("Linear Regression Coefficient")
plt.ylabel("Feature")
plt.title("Top 10 Feature Coefficients")
plt.tight_layout()

plt.savefig("static/graphs/feature_coefficients.png")
plt.close()

# 11. Save trained model
joblib.dump(model, "insurance_model.joblib")

# 12. Final messages
print("\nGraphs saved successfully!")
print("1. static/graphs/actual_vs_predicted.png")
print("2. static/graphs/feature_coefficients.png")
print("Model saved as insurance_model.joblib")


