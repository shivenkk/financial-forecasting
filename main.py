import torch
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import MinMaxScaler
from lstm_model import LSTMModel
from dotenv import load_dotenv
import os
import requests
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

app = FastAPI(title="Financial Market Forecasting API")

# Global variables for models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = None
garch_model = None
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

def load_models():
    global lstm_model, garch_model
    
    # Load LSTM model
    try:
        lstm_model = LSTMModel().to(device)
        lstm_model.load_state_dict(torch.load('lstm_model.pth', map_location=device, weights_only=True))
        lstm_model.eval()
        print("✅ LSTM model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading LSTM model: {e}")
        lstm_model = None
    
    # Load GARCH model
    try:
        with open('garch_model.pkl', 'rb') as f:
            garch_model = pickle.load(f)
        print("✅ GARCH model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading GARCH model: {e}")
        garch_model = None

# Load models on startup
load_models()

def fetch_latest_data(ticker, days=150):
    """Fetch latest data from Polygon.io"""
    if not POLYGON_API_KEY:
        return None
    
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'results' not in data:
            return None
        
        df = pd.DataFrame(data['results'])
        df['Date'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        })
        df['Adj Close'] = df['Close']
        df['Returns'] = df['Close'].pct_change()
        
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Returns']]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

@app.get("/")
async def root():
    return {
        "message": "Financial Market Forecasting API",
        "endpoints": {
            "/docs": "Interactive API documentation",
            "/health": "Check API health and model status",
            "/predict": "Get price and volatility predictions"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "lstm_loaded": lstm_model is not None,
        "garch_loaded": garch_model is not None,
        "polygon_api_configured": POLYGON_API_KEY is not None
    }

@app.get("/predict")
async def predict(ticker: str = 'AAPL', days: int = 5, use_latest: bool = True):
    """
    Predict stock price and volatility
    
    Parameters:
    - ticker: Stock symbol (default: AAPL)
    - days: Number of days to forecast volatility (default: 5)
    - use_latest: Whether to fetch latest data from Polygon.io (default: True)
    """
    
    if lstm_model is None or garch_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Please train models first.")
    
    try:
        # Get data
        if use_latest and POLYGON_API_KEY:
            print(f"Fetching latest data for {ticker}...")
            data = fetch_latest_data(ticker)
            if data is None:
                print("Failed to fetch latest data, using saved data...")
                data = pd.read_csv('aapl_data.csv')
                data['Date'] = pd.to_datetime(data['Date'])
        else:
            data = pd.read_csv('aapl_data.csv')
            data['Date'] = pd.to_datetime(data['Date'])
        
        # Sort by date
        data = data.sort_values('Date')
        
        # Get prices and returns
        prices = data['Close'].values
        returns = data['Returns'].dropna().values
        
        print(f"Data points: {len(prices)}")
        
        # Prepare data for LSTM
        sequence_length = 100
        if len(prices) < sequence_length:
            sequence_length = max(10, len(prices) - 1)
            print(f"Adjusted sequence length to {sequence_length}")
        
        # Normalize prices for LSTM
        scaler = MinMaxScaler()
        prices_normalized = scaler.fit_transform(prices.reshape(-1, 1))
        
        # Use last 'sequence_length' days for prediction
        X = prices_normalized[-sequence_length:].reshape(1, sequence_length, 1)
        X_tensor = torch.FloatTensor(X).to(device)
        
        # LSTM prediction
        with torch.no_grad():
            lstm_pred_normalized = lstm_model(X_tensor).cpu().numpy()
        lstm_pred = scaler.inverse_transform(lstm_pred_normalized)[0][0]
        
        # Current price
        current_price = float(prices[-1])
        
        # Price change prediction
        price_change = lstm_pred - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # GARCH prediction for volatility
        recent_returns = returns[-100:] if len(returns) >= 100 else returns
        recent_returns_pct = recent_returns * 100
        
        # Forecast volatility
        try:
            forecast = garch_model.forecast(horizon=days, start=len(recent_returns_pct))
            volatility = np.sqrt(forecast.variance.values[-1, :])
            volatility_annual = volatility * np.sqrt(252)
        except:
            # Fallback volatility calculation
            volatility = np.std(recent_returns_pct) * np.ones(days)
            volatility_annual = volatility * np.sqrt(252)
        
        # Calculate confidence intervals
        confidence_intervals = []
        for i in range(1, days + 1):
            daily_vol = volatility[i-1] / 100
            upper = current_price * np.exp(i * price_change / current_price / days + 1.96 * daily_vol * np.sqrt(i))
            lower = current_price * np.exp(i * price_change / current_price / days - 1.96 * daily_vol * np.sqrt(i))
            confidence_intervals.append({
                "day": i,
                "upper": round(float(upper), 2),
                "lower": round(float(lower), 2)
            })
        
        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "predicted_price": round(float(lstm_pred), 2),
            "price_change": round(float(price_change), 2),
            "price_change_percent": round(float(price_change_pct), 2),
            "forecast_days": days,
            "daily_volatility_percent": [round(float(v), 4) for v in volatility],
            "annualized_volatility_percent": [round(float(v), 2) for v in volatility_annual],
            "confidence_intervals_95": confidence_intervals,
            "data_points_used": len(prices),
            "last_update": data['Date'].max().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/backtest")
async def backtest(test_size: int = 20):
    """
    Simple backtest of the model on historical data
    """
    if lstm_model is None:
        raise HTTPException(status_code=500, detail="LSTM model not loaded")
    
    try:
        # Load data
        data = pd.read_csv('aapl_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')
        
        prices = data['Close'].values
        
        if len(prices) < 120:
            raise HTTPException(status_code=400, detail="Not enough data for backtesting")
        
        # Normalize all prices
        scaler = MinMaxScaler()
        prices_normalized = scaler.fit_transform(prices.reshape(-1, 1))
        
        predictions = []
        actuals = []
        
        # Make predictions for the last 'test_size' days
        sequence_length = 100
        for i in range(test_size):
            idx = len(prices) - test_size + i
            if idx >= sequence_length:
                X = prices_normalized[idx-sequence_length:idx].reshape(1, sequence_length, 1)
                X_tensor = torch.FloatTensor(X).to(device)
                
                with torch.no_grad():
                    pred_normalized = lstm_model(X_tensor).cpu().numpy()
                pred = scaler.inverse_transform(pred_normalized)[0][0]
                
                predictions.append(float(pred))
                if i < test_size - 1:  # We don't have actual for the last prediction
                    actuals.append(float(prices[idx]))
        
        # Calculate metrics
        if len(actuals) > 0:
            mse = np.mean([(p - a) ** 2 for p, a in zip(predictions[:-1], actuals)])
            rmse = np.sqrt(mse)
            mae = np.mean([abs(p - a) for p, a in zip(predictions[:-1], actuals)])
            mape = np.mean([abs(p - a) / a * 100 for p, a in zip(predictions[:-1], actuals)])
        else:
            rmse = mae = mape = 0
        
        return {
            "test_size": test_size,
            "predictions": predictions,
            "actuals": actuals,
            "metrics": {
                "rmse": round(float(rmse), 4),
                "mae": round(float(mae), 4),
                "mape": round(float(mape), 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Financial Forecasting API...")
    print(f"📊 Using device: {device}")
    print(f"🔗 API documentation will be available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
