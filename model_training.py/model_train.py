import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load your dataset
df = pd.read_csv("crop_yield_predict.csv")

# Feature columns based on actual CSV headers
X = df[['rainfall_mm', 'soil_quality_index', 'farm_size_hectares', 'sunlight_hours', 'fertilizer_kg']]
y = df['crop_yield']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save model

joblib.dump(model, 'flask_app/crop_yield_model.pkl')

print("Model training completed and saved!")
