import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set matplotlib to use Agg backend for server environments
plt.switch_backend('Agg')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def create_visualizations(df, model, X_test, y_test):
    """Generate visualizations during training"""
    print("\nGenerating visualizations...")
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Calculate price per sqft
    df['price_per_sqft'] = df['price'] / df['area_sqft']
    
    # Get predictions for visualization
    y_pred = model.predict(df[['area_sqft']])
    r2 = r2_score(df['price'], y_pred)
    
    # 1. Price Distribution Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Real Estate Price Analysis', fontsize=16, fontweight='bold')
    
    # Price histogram
    axes[0,0].hist(df['price'], bins=25, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(df['price'].mean(), color='red', linestyle='--', 
                      label=f'Mean: ₹{df["price"].mean()/100000:.1f}L')
    axes[0,0].axvline(df['price'].median(), color='orange', linestyle='--', 
                      label=f'Median: ₹{df["price"].median()/100000:.1f}L')
    axes[0,0].set_xlabel('Price (₹)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Price Distribution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    
    # Area distribution
    axes[0,1].hist(df['area_sqft'], bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].axvline(df['area_sqft'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["area_sqft"].mean():.0f} sqft')
    axes[0,1].set_xlabel('Area (sqft)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Area Distribution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Area vs Price scatter
    axes[1,0].scatter(df['area_sqft'], df['price'], alpha=0.6, color='blue', s=50)
    axes[1,0].plot(df['area_sqft'], y_pred, color='red', linewidth=2, 
                   label=f'Linear Fit (R² = {r2:.3f})')
    axes[1,0].set_xlabel('Area (sqft)')
    axes[1,0].set_ylabel('Price (₹)')
    axes[1,0].set_title('Area vs Price Relationship')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    
    # Price per sqft distribution
    axes[1,1].hist(df['price_per_sqft'], bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1,1].axvline(df['price_per_sqft'].mean(), color='red', linestyle='--', 
                      label=f'Mean: ₹{df["price_per_sqft"].mean():.0f}/sqft')
    axes[1,1].set_xlabel('Price per sqft (₹)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Price per Sqft Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Model Performance Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # Training vs Test Performance
    X_train, X_test_split, y_train, y_test_split = train_test_split(
        df[['area_sqft']], df['price'], test_size=0.2, random_state=42
    )
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test_split)
    
    # Predicted vs Actual scatter
    axes[0].scatter(y_test_split, y_pred_test, alpha=0.6, color='blue', s=50)
    axes[0].plot([y_test_split.min(), y_test_split.max()], 
                 [y_test_split.min(), y_test_split.max()], 'r--', linewidth=2)
    axes[0].set_xlabel('Actual Price (₹)')
    axes[0].set_ylabel('Predicted Price (₹)')
    axes[0].set_title(f'Predicted vs Actual (R² = {r2_score(y_test_split, y_pred_test):.3f})')
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    
    # Residuals plot
    residuals = y_test_split - y_pred_test
    axes[1].scatter(y_pred_test, residuals, alpha=0.6, color='green')
    axes[1].axhline(y=0, color='red', linestyle='--')
    axes[1].set_xlabel('Predicted Price (₹)')
    axes[1].set_ylabel('Residuals (₹)')
    axes[1].set_title('Residual Analysis')
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    
    plt.tight_layout()
    plt.savefig('static/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Visualizations saved to static/ directory")
    print("  - analysis_dashboard.png")
    print("  - model_performance.png")

print("Loading data...")
df = pd.read_csv("data.csv")

print(f"\nDataset Info:")
print(f"Total records: {len(df)}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nData statistics:")
print(df.describe())

# Data cleaning
if df.isnull().sum().sum() > 0:
    print("\nRemoving rows with missing values...")
    df = df.dropna()

# Remove outliers using IQR method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]

# Ensure positive values
df = df[(df['area_sqft'] > 0) & (df['price'] > 0)]

print(f"\nRecords after cleaning: {len(df)}")

# Prepare features and target
X = df[['area_sqft']]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
print("\nTraining model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
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

# Generate visualizations
create_visualizations(df, model, X_test, y_test)

# Save model with additional data
model_data = {
    'model': model,
    'min_area': float(df['area_sqft'].min()),
    'max_area': float(df['area_sqft'].max()),
    'mean_price': float(df['price'].mean()),
    'r2_score': float(r2_score(y_test, y_pred_test)),
    'mean_price_per_sqft': float(df['price'].mean() / df['area_sqft'].mean()),
    'dataset_size': len(df)
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\nModel saved successfully!")
print("="*50)