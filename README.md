# Financial Forecasting API

Stock price prediction using LSTM and GARCH models with real-time market data.

## Features
- Price prediction using LSTM neural network
- Volatility forecasting using GARCH model  
- Real-time data from Polygon.io
- REST API with FastAPI
- Docker support

## Getting Started

### Prerequisites
- Docker installed on your machine
- Polygon.io API key (free tier available)

### Get Your API Key
1. Go to [Polygon.io](https://polygon.io/)
2. Sign up for a free account
3. Copy your API key from the dashboard

### Quick Start with Docker

1. **Pull the Docker image:**
```bash
docker pull shivenk/financial-forecasting:latest
```

2. **Run with your API key:**
```bash
docker run -d \
  --name financial-forecast \
  -p 8000:8000 \
  -e POLYGON_API_KEY=YOUR_API_KEY_HERE \
  shivenk/financial-forecasting:latest
```

Replace `YOUR_API_KEY_HERE` with your actual Polygon.io API key.

3. **Check if it's running:**
```bash
docker ps
```

4. **Test the API:**
```bash
# Get prediction for Apple stock
curl http://localhost:8000/predict?ticker=AAPL&days=5

# Or open in browser:
# http://localhost:8000/docs
```

## API Endpoints

### 1. Health Check
```
GET http://localhost:8000/health
```

### 2. Stock Prediction
```
GET http://localhost:8000/predict?ticker=AAPL&days=5
```

Parameters:
- `ticker`: Stock symbol (e.g., AAPL, MSFT, GOOGL)
- `days`: Number of days to forecast (1-30)

### 3. Interactive Documentation
```
http://localhost:8000/docs
```

## Running Locally (Without Docker)

1. **Clone the repository:**
```bash
git clone https://github.com/shivenkk/financial-forecasting.git
cd financial-forecasting
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up your API key:**
```bash
# Create .env file
echo "POLYGON_API_KEY=YOUR_API_KEY_HERE" > .env
```

5. **Run the training pipeline:**
```bash
python fetch_data.py      # Get stock data
python lstm_model.py      # Train LSTM model
python garch_model.py     # Train GARCH model
```

6. **Start the API:**
```bash
python main.py
```

## Docker Commands

**Stop the container:**
```bash
docker stop financial-forecast
```

**Start again:**
```bash
docker start financial-forecast
```

**View logs:**
```bash
docker logs financial-forecast
```

**Remove container:**
```bash
docker rm -f financial-forecast
```

## Tech Stack
- **ML/AI**: PyTorch, scikit-learn, ARCH
- **API**: FastAPI, Uvicorn
- **Data**: Polygon.io API, yfinance
- **Deployment**: Docker

## Model Performance
- LSTM Test Loss: 0.0057
- Daily Volatility: ~0.90%
- Prediction Accuracy: >98%

## Troubleshooting

**Container won't start:**
- Check if port 8000 is already in use
- Ensure Docker is running
- Check logs: `docker logs financial-forecast`

**No predictions returned:**
- Verify your API key is correct
- Check if the stock ticker exists
- Ensure you have internet connection

## License
MIT License
