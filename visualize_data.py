import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("Loading data...")
    df = pd.read_csv("data.csv")
    
    # Remove any missing values
    df = df.dropna()
    
    # Remove outliers using IQR method
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]
    
    # Ensure positive values
    df = df[(df['area_sqft'] > 0) & (df['price'] > 0)]
    
    print(f"Dataset loaded: {len(df)} records")
    return df

def plot_price_distribution(df):
    """Create visualization for price distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Price Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram with density curve
    axes[0,0].hist(df['price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(df['price'].mean(), color='red', linestyle='--', 
                      label=f'Mean: ₹{df["price"].mean()/100000:.1f}L')
    axes[0,0].axvline(df['price'].median(), color='orange', linestyle='--', 
                      label=f'Median: ₹{df["price"].median()/100000:.1f}L')
    axes[0,0].set_xlabel('Price (₹)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Price Distribution (Histogram)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Format x-axis to show prices in lakhs
    axes[0,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    
    # 2. Box plot
    axes[0,1].boxplot(df['price'], patch_artist=True, 
                      boxprops=dict(facecolor='lightgreen', alpha=0.7))
    axes[0,1].set_ylabel('Price (₹)')
    axes[0,1].set_title('Price Distribution (Box Plot)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    
    # 3. Price per sqft distribution
    df['price_per_sqft'] = df['price'] / df['area_sqft']
    axes[1,0].hist(df['price_per_sqft'], bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1,0].axvline(df['price_per_sqft'].mean(), color='red', linestyle='--', 
                      label=f'Mean: ₹{df["price_per_sqft"].mean():.0f}/sqft')
    axes[1,0].set_xlabel('Price per Square Foot (₹)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Price per Square Foot Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Summary statistics
    axes[1,1].axis('off')
    stats_text = f"""
    PRICE STATISTICS
    
    Count: {len(df):,} properties
    Mean: ₹{df['price'].mean()/100000:.2f} Lakhs
    Median: ₹{df['price'].median()/100000:.2f} Lakhs
    Std Dev: ₹{df['price'].std()/100000:.2f} Lakhs
    
    Min: ₹{df['price'].min()/100000:.2f} Lakhs
    Max: ₹{df['price'].max()/100000:.2f} Lakhs
    
    PRICE PER SQFT STATISTICS
    
    Mean: ₹{df['price_per_sqft'].mean():.0f}/sqft
    Median: ₹{df['price_per_sqft'].median():.0f}/sqft
    Min: ₹{df['price_per_sqft'].min():.0f}/sqft
    Max: ₹{df['price_per_sqft'].max():.0f}/sqft
    """
    axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes, 
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('price_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Price distribution plot saved as 'price_distribution.png'")

def plot_area_price_relationship(df):
    """Create visualization for area vs price relationship"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Area vs Price Relationship Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot with regression line
    X = df[['area_sqft']]
    y = df['price']
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    axes[0,0].scatter(df['area_sqft'], df['price'], alpha=0.6, color='blue', s=50)
    axes[0,0].plot(df['area_sqft'], y_pred, color='red', linewidth=2, 
                   label=f'Linear Fit (R² = {r2:.3f})')
    axes[0,0].set_xlabel('Area (Square Feet)')
    axes[0,0].set_ylabel('Price (₹)')
    axes[0,0].set_title('Area vs Price Scatter Plot')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    
    # 2. Residual plot
    residuals = y - y_pred
    axes[0,1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0,1].axhline(y=0, color='red', linestyle='--')
    axes[0,1].set_xlabel('Predicted Price (₹)')
    axes[0,1].set_ylabel('Residuals (₹)')
    axes[0,1].set_title('Residual Plot')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    
    # 3. Area distribution
    axes[1,0].hist(df['area_sqft'], bins=25, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].axvline(df['area_sqft'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["area_sqft"].mean():.0f} sqft')
    axes[1,0].axvline(df['area_sqft'].median(), color='blue', linestyle='--', 
                      label=f'Median: {df["area_sqft"].median():.0f} sqft')
    axes[1,0].set_xlabel('Area (Square Feet)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Area Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Model statistics and insights
    axes[1,1].axis('off')
    slope = model.coef_[0]
    intercept = model.intercept_
    
    model_text = f"""
    LINEAR REGRESSION MODEL
    
    Equation: Price = {slope:.2f} × Area + {intercept:.0f}
    
    Model Performance:
    R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)
    
    Interpretation:
    • Price increases by ₹{slope:.0f} per sqft
    • Base price (intercept): ₹{intercept/100000:.2f} Lakhs
    
    AREA STATISTICS
    
    Count: {len(df):,} properties
    Mean: {df['area_sqft'].mean():.0f} sqft
    Median: {df['area_sqft'].median():.0f} sqft
    Min: {df['area_sqft'].min():.0f} sqft
    Max: {df['area_sqft'].max():.0f} sqft
    
    Range: {df['area_sqft'].max() - df['area_sqft'].min():.0f} sqft
    """
    axes[1,1].text(0.1, 0.9, model_text, transform=axes[1,1].transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('area_price_relationship.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Area vs Price relationship plot saved as 'area_price_relationship.png'")

def create_combined_dashboard(df):
    """Create a combined dashboard with key insights"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('Flat Price Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Price histogram (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['price'], bins=25, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Price Distribution', fontweight='bold')
    ax1.set_xlabel('Price (₹)')
    ax1.set_ylabel('Frequency')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    ax1.grid(True, alpha=0.3)
    
    # 2. Area histogram (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df['area_sqft'], bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Area Distribution', fontweight='bold')
    ax2.set_xlabel('Area (sqft)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # 3. Price per sqft histogram (top-right)
    df['price_per_sqft'] = df['price'] / df['area_sqft']
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df['price_per_sqft'], bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_title('Price per Sqft Distribution', fontweight='bold')
    ax3.set_xlabel('Price per sqft (₹)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 4. Main scatter plot (middle span)
    ax4 = fig.add_subplot(gs[1, :])
    
    # Fit model for the scatter plot
    X = df[['area_sqft']]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    ax4.scatter(df['area_sqft'], df['price'], alpha=0.6, color='blue', s=60)
    ax4.plot(df['area_sqft'], y_pred, color='red', linewidth=3, 
             label=f'Linear Regression (R² = {r2:.3f})')
    ax4.set_title('Area vs Price Relationship', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Area (Square Feet)')
    ax4.set_ylabel('Price (₹)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
    
    # 5. Summary statistics (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    summary_text = f"""
DATASET SUMMARY

Total Properties: {len(df):,}

PRICE STATISTICS
Mean: ₹{df['price'].mean()/100000:.1f}L
Median: ₹{df['price'].median()/100000:.1f}L
Range: ₹{df['price'].min()/100000:.1f}L - ₹{df['price'].max()/100000:.1f}L

AREA STATISTICS
Mean: {df['area_sqft'].mean():.0f} sqft
Median: {df['area_sqft'].median():.0f} sqft
Range: {df['area_sqft'].min():.0f} - {df['area_sqft'].max():.0f} sqft
    """
    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightblue", alpha=0.8))
    
    # 6. Model insights (bottom-middle)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    slope = model.coef_[0]
    intercept = model.intercept_
    model_text = f"""
MODEL INSIGHTS

Linear Equation:
Price = {slope:.0f} × Area + {intercept:.0f}

Performance:
R² Score: {r2:.4f} ({r2*100:.1f}%)

Key Finding:
Every additional sqft adds
₹{slope:.0f} to the price

Average Price/Sqft:
₹{df['price_per_sqft'].mean():.0f}
    """
    ax6.text(0.1, 0.9, model_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightgreen", alpha=0.8))
    
    # 7. Price ranges (bottom-right)
    ax7 = fig.add_subplot(gs[2, 2])
    
    # Create price range categories
    df['price_category'] = pd.cut(df['price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    price_counts = df['price_category'].value_counts()
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    wedges, texts, autotexts = ax7.pie(price_counts.values, labels=price_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax7.set_title('Price Range Distribution', fontweight='bold')
    
    plt.savefig('dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Combined dashboard saved as 'dashboard.png'")

def main():
    """Main function to run all visualizations"""
    # Load data
    df = load_and_prepare_data()
    
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    # Create visualizations
    print("\n1. Creating price distribution plots...")
    plot_price_distribution(df)
    
    print("\n2. Creating area vs price relationship plots...")
    plot_area_price_relationship(df)
    
    print("\n3. Creating combined dashboard...")
    create_combined_dashboard(df)
    
    print("\n" + "="*50)
    print("VISUALIZATION COMPLETE!")
    print("="*50)
    print("\nGenerated files:")
    print("• price_distribution.png")
    print("• area_price_relationship.png") 
    print("• dashboard.png")
    print("\nAll visualizations have been saved to the current directory.")

if __name__ == "__main__":
    main()