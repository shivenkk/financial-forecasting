import torch
import torch.nn as nn
import numpy as np
from preprocess_data import preprocess_data

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm(X_train, y_train, epochs=50, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    
    # Adjust batch size if necessary
    batch_size = min(batch_size, len(X_train))
    
    print(f"Training LSTM model...")
    print(f"Train samples: {len(X_train)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    torch.save(model.state_dict(), 'lstm_model.pth')
    print("LSTM model trained and saved to lstm_model.pth")
    return model

if __name__ == "__main__":
    try:
        # Load and preprocess data
        X_train, y_train, X_test, y_test, _, _, _ = preprocess_data('aapl_data.csv')
        
        # Check if we have enough data
        if len(X_train) == 0:
            print("Error: No training data available. Please check your data file.")
        else:
            # Train model
            model = train_lstm(X_train, y_train)
            
            # Optional: Evaluate on test set
            if len(X_test) > 0:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test).to(device)
                    y_test_tensor = torch.FloatTensor(y_test).to(device)
                    test_predictions = model(X_test_tensor)
                    test_loss = nn.MSELoss()(test_predictions, y_test_tensor)
                    print(f"\nTest Loss: {test_loss.item():.6f}")
    
    except Exception as e:
        print(f"Error training LSTM model: {e}")
        import traceback
        traceback.print_exc()
