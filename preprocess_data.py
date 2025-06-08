import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path, sequence_length=100):
    # Load data
    data = pd.read_csv(file_path)
    print(f"Loaded data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Ensure Date is datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Sort by date
    data = data.sort_values('Date')
    
    # Check if we have enough data
    if len(data) < sequence_length + 1:
        print(f"Warning: Data has only {len(data)} rows, but sequence_length is {sequence_length}")
        print(f"Adjusting sequence_length to {max(10, len(data) // 2)}")
        sequence_length = max(10, len(data) // 2)
    
    # Ensure 'Close' column exists and is numeric
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])
    
    # Handle 'Returns' column
    if 'Returns' not in data.columns:
        data['Returns'] = data['Close'].pct_change()
    else:
        data['Returns'] = pd.to_numeric(data['Returns'], errors='coerce')
    
    # Drop the first row with NaN return
    data = data.dropna(subset=['Returns'])
    
    # Select features (Close price and Returns)
    prices = data['Close'].values.reshape(-1, 1)
    returns = data['Returns'].values
    
    print(f"Number of prices: {len(prices)}")
    print(f"Number of returns: {len(returns)}")
    print(f"First 5 Close prices: {prices[:5].flatten()}")
    print(f"First 5 Returns: {returns[:5]}")
    
    # Normalize prices
    scaler = MinMaxScaler()
    prices_normalized = scaler.fit_transform(prices)
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(prices_normalized) - sequence_length):
        X.append(prices_normalized[i:i+sequence_length])
        y.append(prices_normalized[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} sequences")
    
    if len(X) == 0:
        raise ValueError(f"Not enough data to create sequences. Need at least {sequence_length + 1} data points.")
    
    # Split into train and test (80-20)
    train_size = int(0.8 * len(X))
    train_size = max(1, train_size)  # Ensure at least 1 training sample
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # For returns, we need to align them properly
    # Skip the first 'sequence_length' returns since they were used to create X[0]
    aligned_returns = returns[sequence_length:]
    returns_train = aligned_returns[:train_size]
    returns_test = aligned_returns[train_size:] if train_size < len(aligned_returns) else np.array([])
    
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Returns train size: {len(returns_train)}")
    print(f"Returns test size: {len(returns_test)}")
    
    return X_train, y_train, X_test, y_test, returns_train, returns_test, scaler

if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test, returns_train, returns_test, scaler = preprocess_data('aapl_data.csv')
        print("\nData preprocessed successfully!")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        import traceback
        traceback.print_exc()
