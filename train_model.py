import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import numpy as np


print("Loading data...")
df = pd.read_csv("data.csv")

print(f"\nDataset Info:")
print(f"Total records: {len(df)}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nData statistics:")
print(df.describe())


if df.isnull().sum().sum() > 0:
    print("\nRemoving rows with missing values...")
    df = df.dropna()


Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]


df = df[(df['area_sqft'] > 0) & (df['price'] > 0)]

print(f"\nRecords after cleaning: {len(df)}")


X = df[['area_sqft']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("\nTraining model...")
model = LinearRegression()
model.fit(X_train, y_train)


y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"\nTraining Set:")
print(f"  R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"  MAE: ₹{mean_absolute_error(y_train, y_pred_train):,.0f}")
print(f"  RMSE: ₹{np.sqrt(mean_squared_error(y_train, y_pred_train)):,.0f}")

print(f"\nTest Set:")
print(f"  R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"  MAE: ₹{mean_absolute_error(y_test, y_pred_test):,.0f}")
print(f"  RMSE: ₹{np.sqrt(mean_squared_error(y_test, y_pred_test)):,.0f}")

print(f"\nModel Coefficients:")
print(f"  Price per sqft: ₹{model.coef_[0]:,.2f}")
print(f"  Intercept: ₹{model.intercept_:,.2f}")

#
model_data = {
    'model': model,
    'min_area': float(df['area_sqft'].min()),
    'max_area': float(df['area_sqft'].max()),
    'mean_price': float(df['price'].mean()),
    'r2_score': float(r2_score(y_test, y_pred_test))
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\nModel saved successfully!")
print("="*50)