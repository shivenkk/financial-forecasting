import pandas as pd
import requests
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()
API_KEY = os.getenv('POLYGON_API_KEY')

def fetch_stock_data(ticker='AAPL', start_date='2020-01-01', end_date='2023-12-31'):
    """Fetch stock data from Polygon.io"""
    print(f"Fetching {ticker} data from Polygon.io...")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if 'results' not in data:
        print(f"API Response: {data}")
        raise ValueError("Failed to fetch data from Polygon.io")
    
    print(f"Fetched {len(data['results'])} days of data")
    
    # Convert to DataFrame
    df = pd.DataFrame(data['results'])
    
    # Convert timestamp to datetime
    df['Date'] = pd.to_datetime(df['t'], unit='ms')
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'o': 'Open',
        'h': 'High',
        'l': 'Low',
        'c': 'Close',
        'v': 'Volume',
        'vw': 'VWAP',  # Volume weighted average price if available
        'n': 'Transactions'  # Number of transactions if available
    })
    
    # Set Date as index
    df = df.set_index('Date')
    
    # Select and order columns
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[columns]
    
    # Add Adjusted Close (same as Close for Polygon data)
    df['Adj Close'] = df['Close']
    
    # Calculate Returns
    df['Returns'] = df['Close'].pct_change()
    
    # Reset index to have Date as a column (required by other scripts)
    df = df.reset_index()
    
    return df

def main():
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    try:
        # Fetch data
        df = fetch_stock_data(ticker, start_date, end_date)
        
        # Save to CSV
        df.to_csv('aapl_data.csv', index=False)
        
        print(f"\n✅ Data saved to aapl_data.csv")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Display summary statistics
        print("\nSummary Statistics:")
        print(f"Starting Price: ${df.iloc[0]['Close']:.2f}")
        print(f"Ending Price: ${df.iloc[-1]['Close']:.2f}")
        total_return = ((df.iloc[-1]['Close'] / df.iloc[0]['Close']) - 1) * 100
        print(f"Total Return: {total_return:.1f}%")
        
        # Calculate metrics
        returns = df['Returns'].dropna()
        print(f"Average Daily Return: {returns.mean() * 100:.3f}%")
        print(f"Daily Volatility: {returns.std() * 100:.2f}%")
        print(f"Annualized Volatility: {returns.std() * np.sqrt(252) * 100:.1f}%")
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nLast 5 rows:")
        print(df.tail())
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have set POLYGON_API_KEY in your .env file")
        print("Get a free API key from: https://polygon.io/")

if __name__ == "__main__":
    main()
