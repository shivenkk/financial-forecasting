import pickle
import numpy as np
from arch import arch_model
from preprocess_data import preprocess_data

def train_garch(returns_train):
    # Remove any NaN values
    returns_train = returns_train[~np.isnan(returns_train)]
    
    if len(returns_train) < 10:
        print(f"Warning: Only {len(returns_train)} returns available for GARCH training")
        print("GARCH models typically need more data for reliable estimates")
    
    # Convert to percentage returns for GARCH
    returns_pct = returns_train * 100
    
    print(f"Training GARCH model with {len(returns_pct)} returns")
    print(f"Mean return: {np.mean(returns_pct):.4f}%")
    print(f"Std deviation: {np.std(returns_pct):.4f}%")
    
    # Fit GARCH(1,1) model
    model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='normal')
    
    try:
        fitted_model = model.fit(disp='off', show_warning=False)
        print("\nGARCH model fitted successfully")
        print(fitted_model.summary())
    except Exception as e:
        print(f"Warning during GARCH fitting: {e}")
        # Try with different options
        print("Trying alternative GARCH specification...")
        model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='normal', mean='Zero')
        fitted_model = model.fit(disp='off', show_warning=False, options={'maxiter': 200})
    
    # Save model
    with open('garch_model.pkl', 'wb') as f:
        pickle.dump(fitted_model, f)
    
    print("\nGARCH model saved to garch_model.pkl")
    
    # Show volatility forecast for next 5 days
    try:
        forecast = fitted_model.forecast(horizon=5)
        volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
        print(f"\nVolatility forecast for next 5 days (%):")
        for i, vol in enumerate(volatility_forecast):
            print(f"  Day {i+1}: {vol:.4f}%")
    except Exception as e:
        print(f"Could not generate forecast: {e}")
    
    return fitted_model

if __name__ == "__main__":
    try:
        # Load and preprocess data
        _, _, _, _, returns_train, _, _ = preprocess_data('aapl_data.csv')
        
        if len(returns_train) == 0:
            print("Error: No returns data available for GARCH training")
            print("Loading raw data to extract returns...")
            
            # Try loading directly from CSV
            import pandas as pd
            data = pd.read_csv('aapl_data.csv')
            if 'Returns' in data.columns:
                returns = data['Returns'].dropna().values
                if len(returns) > 20:
                    # Use 80% for training
                    train_size = int(0.8 * len(returns))
                    returns_train = returns[:train_size]
                else:
                    returns_train = returns
        
        if len(returns_train) > 0:
            model = train_garch(returns_train)
        else:
            print("Error: Could not extract returns for GARCH training")
            
    except Exception as e:
        print(f"Error training GARCH model: {e}")
        import traceback
        traceback.print_exc()
