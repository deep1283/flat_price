from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pickle
from typing import Optional
import os

app = FastAPI(title="Flat Price Predictor")

# Mount static files directory
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and metadata
try:
    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    min_area = model_data['min_area']
    max_area = model_data['max_area']
    r2_score = model_data.get('r2_score', 0)
    mean_price_per_sqft = model_data.get('mean_price_per_sqft', 0)
    dataset_size = model_data.get('dataset_size', 0)
    
    print(f"Model loaded successfully!")
    print(f"Valid area range: {min_area:.0f} - {max_area:.0f} sqft")
    print(f"Model R² Score: {r2_score:.4f}")
    print(f"Dataset size: {dataset_size} properties")
except FileNotFoundError:
    print("Error: model.pkl not found. Please run train_model.py first.")
    raise
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.get("/", response_class=HTMLResponse)
async def form():
    # Check if visualizations exist
    dashboard_exists = os.path.exists("static/analysis_dashboard.png")
    performance_exists = os.path.exists("static/model_performance.png")
    
    visualization_section = ""
    if dashboard_exists or performance_exists:
        visualization_section = f"""
        <div class="nav-buttons">
            <a href="/analytics" class="nav-button">📊 View Analytics</a>
            <a href="/model-performance" class="nav-button">🎯 Model Performance</a>
        </div>
        """
    
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Flat Price Predictor</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 15px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    max-width: 450px;
                    width: 100%;
                }}
                h2 {{
                    color: #333;
                    margin-bottom: 10px;
                    font-size: 28px;
                }}
                .subtitle {{
                    color: #666;
                    margin-bottom: 30px;
                    font-size: 14px;
                }}
                .nav-buttons {{
                    display: flex;
                    gap: 10px;
                    margin-bottom: 25px;
                    flex-wrap: wrap;
                }}
                .nav-button {{
                    flex: 1;
                    padding: 10px 15px;
                    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 6px;
                    font-size: 13px;
                    font-weight: 600;
                    text-align: center;
                    transition: transform 0.2s, box-shadow 0.2s;
                    min-width: 120px;
                }}
                .nav-button:hover {{
                    transform: translateY(-1px);
                    box-shadow: 0 3px 15px rgba(40, 167, 69, 0.4);
                }}
                label {{
                    display: block;
                    margin-bottom: 8px;
                    color: #555;
                    font-weight: 500;
                }}
                input[type="number"] {{
                    width: 100%;
                    padding: 12px 15px;
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    font-size: 16px;
                    transition: border-color 0.3s;
                    margin-bottom: 20px;
                }}
                input[type="number"]:focus {{
                    outline: none;
                    border-color: #667eea;
                }}
                button {{
                    width: 100%;
                    padding: 14px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: transform 0.2s, box-shadow 0.2s;
                }}
                button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
                }}
                button:active {{
                    transform: translateY(0);
                }}
                .info {{
                    margin-top: 20px;
                    padding: 15px;
                    background: #f5f5f5;
                    border-radius: 8px;
                    font-size: 13px;
                    color: #666;
                }}
                .info-item {{
                    margin: 5px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>🏠 Flat Price Predictor</h2>
                <p class="subtitle">Get instant price estimates based on area</p>
                {visualization_section}
                <form method="post">
                    <label for="area">Area (Square Feet)</label>
                    <input 
                        type="number" 
                        id="area"
                        name="area" 
                        placeholder="e.g., 1000"
                        min="{int(min_area)}"
                        max="{int(max_area)}"
                        required 
                    />
                    <button type="submit">Predict Price</button>
                </form>
                <div class="info">
                    <div class="info-item">📏 Valid range: {int(min_area):,} - {int(max_area):,} sqft</div>
                    <div class="info-item">📊 Model accuracy (R²): {r2_score:.2%}</div>
                    <div class="info-item">🏠 Dataset size: {dataset_size:,} properties</div>
                    <div class="info-item">💰 Avg price/sqft: ₹{mean_price_per_sqft:,.0f}</div>
                </div>
            </div>
        </body>
    </html>
    """

@app.get("/analytics", response_class=HTMLResponse)
async def analytics():
    """Display data analytics dashboard"""
    if not os.path.exists("static/analysis_dashboard.png"):
        return error_page("Analytics Not Available", "Please run train_model.py first to generate analytics.")
    
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Data Analytics - Flat Price Predictor</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 30px 0;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                }}
                .back-button {{
                    display: inline-block;
                    padding: 12px 25px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                    transition: transform 0.2s;
                    margin-bottom: 20px;
                }}
                .back-button:hover {{
                    transform: translateY(-2px);
                }}
                .description {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    color: #555;
                    line-height: 1.6;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/" class="back-button">← Back to Predictor</a>
                <h1>📊 Data Analytics Dashboard</h1>
                
                <div class="description">
                    <p><strong>This dashboard shows comprehensive analysis of the real estate dataset:</strong></p>
                    <ul style="margin: 10px 0 0 20px;">
                        <li><strong>Price Distribution:</strong> Shows how property prices are distributed across the dataset</li>
                        <li><strong>Area Distribution:</strong> Displays the range of property sizes in the dataset</li>
                        <li><strong>Area vs Price Relationship:</strong> Scatter plot showing correlation with regression line</li>
                        <li><strong>Price per Sqft Distribution:</strong> Analysis of cost efficiency patterns</li>
                    </ul>
                </div>
                
                <div class="chart-container">
                    <img src="/static/analysis_dashboard.png" alt="Data Analytics Dashboard" />
                </div>
            </div>
        </body>
    </html>
    """

@app.get("/model-performance", response_class=HTMLResponse)
async def model_performance():
    """Display model performance analysis"""
    if not os.path.exists("static/model_performance.png"):
        return error_page("Performance Analysis Not Available", "Please run train_model.py first to generate performance analysis.")
    
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Model Performance - Flat Price Predictor</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 30px 0;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                }}
                .back-button {{
                    display: inline-block;
                    padding: 12px 25px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                    transition: transform 0.2s;
                    margin-bottom: 20px;
                }}
                .back-button:hover {{
                    transform: translateY(-2px);
                }}
                .description {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    color: #555;
                    line-height: 1.6;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .metric-label {{
                    font-size: 14px;
                    opacity: 0.9;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/" class="back-button">← Back to Predictor</a>
                <h1>🎯 Model Performance Analysis</h1>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{r2_score:.1%}</div>
                        <div class="metric-label">Model Accuracy (R²)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dataset_size:,}</div>
                        <div class="metric-label">Training Samples</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">₹{mean_price_per_sqft:,.0f}</div>
                        <div class="metric-label">Avg Price/Sqft</div>
                    </div>
                </div>
                
                <div class="description">
                    <p><strong>Model Performance Insights:</strong></p>
                    <ul style="margin: 10px 0 0 20px;">
                        <li><strong>Predicted vs Actual:</strong> Shows how well the model predictions match actual prices</li>
                        <li><strong>Residual Analysis:</strong> Displays prediction errors to identify model biases</li>
                        <li><strong>R² Score:</strong> Indicates {r2_score:.1%} of price variance is explained by area</li>
                        <li><strong>Model Type:</strong> Linear Regression optimized for real estate pricing</li>
                    </ul>
                </div>
                
                <div class="chart-container">
                    <img src="/static/model_performance.png" alt="Model Performance Analysis" />
                </div>
            </div>
        </body>
    </html>
    """

@app.post("/", response_class=HTMLResponse)
async def predict(area: int = Form(...)):
    try:
        # Validate input
        if area <= 0:
            raise HTTPException(status_code=400, detail="Area must be positive")
        
        if area < min_area * 0.5 or area > max_area * 1.5:
            return error_page(
                f"Area {area:,} sqft is outside the reliable prediction range",
                f"Please enter a value between {int(min_area):,} and {int(max_area):,} sqft"
            )
        
        # Predict
        predicted_price = model.predict([[area]])[0]
        
        # Warning for values outside training range
        warning = ""
        if area < min_area or area > max_area:
            warning = f"""
                <div class="warning">
                    ⚠️ This area is outside our training data range. 
                    Prediction may be less accurate.
                </div>
            """
        
        return f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Prediction Result</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        padding: 20px;
                    }}
                    .container {{
                        background: white;
                        padding: 40px;
                        border-radius: 15px;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                        max-width: 450px;
                        width: 100%;
                        text-align: center;
                    }}
                    h2 {{
                        color: #333;
                        margin-bottom: 10px;
                        font-size: 24px;
                    }}
                    .price {{
                        font-size: 48px;
                        font-weight: bold;
                        color: #667eea;
                        margin: 20px 0;
                        animation: fadeIn 0.5s;
                    }}
                    .details {{
                        background: #f5f5f5;
                        padding: 20px;
                        border-radius: 8px;
                        margin: 20px 0;
                    }}
                    .detail-item {{
                        display: flex;
                        justify-content: space-between;
                        margin: 10px 0;
                        color: #555;
                    }}
                    .label {{
                        font-weight: 500;
                    }}
                    .value {{
                        font-weight: 600;
                        color: #333;
                    }}
                    .button {{
                        display: inline-block;
                        padding: 14px 30px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 8px;
                        font-weight: 600;
                        transition: transform 0.2s, box-shadow 0.2s;
                        margin: 5px;
                    }}
                    .button:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
                    }}
                    .button.secondary {{
                        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                    }}
                    .warning {{
                        background: #fff3cd;
                        border: 1px solid #ffc107;
                        padding: 15px;
                        border-radius: 8px;
                        margin: 20px 0;
                        color: #856404;
                        font-size: 14px;
                    }}
                    @keyframes fadeIn {{
                        from {{ opacity: 0; transform: translateY(-10px); }}
                        to {{ opacity: 1; transform: translateY(0); }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>💰 Predicted Price</h2>
                    <div class="price">₹{predicted_price:,.0f}</div>
                    {warning}
                    <div class="details">
                        <div class="detail-item">
                            <span class="label">Area:</span>
                            <span class="value">{area:,} sqft</span>
                        </div>
                        <div class="detail-item">
                            <span class="label">Price per sqft:</span>
                            <span class="value">₹{predicted_price/area:,.0f}</span>
                        </div>
                        <div class="detail-item">
                            <span class="label">Model Accuracy:</span>
                            <span class="value">{r2_score:.1%}</span>
                        </div>
                    </div>
                    <a href="/" class="button">← Predict Another</a>
                    <a href="/analytics" class="button secondary">📊 View Analytics</a>
                </div>
            </body>
        </html>
        """
    
    except HTTPException:
        raise
    except Exception as e:
        return error_page("Prediction Error", str(e))

def error_page(title: str, message: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Error</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 15px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    max-width: 450px;
                    width: 100%;
                    text-align: center;
                }}
                .error-icon {{
                    font-size: 64px;
                    margin-bottom: 20px;
                }}
                h2 {{
                    color: #d32f2f;
                    margin-bottom: 15px;
                }}
                p {{
                    color: #666;
                    margin-bottom: 30px;
                    line-height: 1.6;
                }}
                .button {{
                    display: inline-block;
                    padding: 14px 30px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                    transition: transform 0.2s;
                }}
                .button:hover {{
                    transform: translateY(-2px);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="error-icon">⚠️</div>
                <h2>{title}</h2>
                <p>{message}</p>
                <a href="/" class="button">← Go Back</a>
            </div>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)