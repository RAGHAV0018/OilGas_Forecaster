import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime

# -------------------------------------------------------------
# 1. DATA GENERATION (For Phase 1 Proof of Concept)
# In Phase 2, this will be replaced with real PPAC/OPEC CSV data.
# Here we generate synthetic daily oil prices mimicking real trends.
# -------------------------------------------------------------
def generate_mock_oil_data(days=1000):
    print("Generating historical oil price data...")
    np.random.seed(42)
    base_price = 75.0
    prices = [base_price]
    for _ in range(days - 1):
        # Random walk with slight upward drift and some volatility
        volatility = np.random.normal(loc=0.01, scale=1.5)
        new_price = prices[-1] + volatility
        prices.append(max(20.0, new_price)) # prices don't drop below $20
    
    dates = pd.date_range(end=datetime.date.today(), periods=days)
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    df.set_index('Date', inplace=True)
    return df

# -------------------------------------------------------------
# 2. LSTM MODEL DEFINITION
# -------------------------------------------------------------
class OilPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(OilPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully Connected Layer (to output the predicted price)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# -------------------------------------------------------------
# 3. TRAINING & PREDICTION ENGINE
# -------------------------------------------------------------
class PricePredictorEngine:
    def __init__(self, sequence_length=30):
        self.seq_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model = OilPriceLSTM()
        self.criterion = nn.MSELoss() # Mean Squared Error
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def prepare_data(self, df):
        # Normalize the prices to be between -1 and 1 for neural network stability
        price_data = df['Price'].values.reshape(-1, 1)
        normalized_data = self.scaler.fit_transform(price_data)
        
        X, y = [], []
        # Create sequences (e.g., look at past 30 days to predict day 31)
        for i in range(len(normalized_data) - self.seq_length):
            X.append(normalized_data[i : i + self.seq_length])
            y.append(normalized_data[i + self.seq_length])
            
        # Convert to PyTorch tensors
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    def train(self, X, y, epochs=50):
        print(f"Training LSTM Model for {epochs} epochs...")
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        print("Training Complete.\n")

    def predict_future(self, last_sequence, days_to_predict=30):
        self.model.eval()
        predictions = []
        current_seq = last_sequence.clone()
        
        for _ in range(days_to_predict):
            with torch.no_grad():
                pred = self.model(current_seq.unsqueeze(0))
                predictions.append(pred.item())
                # Shift sequence and append new prediction
                current_seq = torch.cat((current_seq[1:], pred))
                
        # Inverse transform to get actual dollar values
        predicted_prices = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predicted_prices.flatten()

# -------------------------------------------------------------
# EXECUTION (Demonstration)
# -------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load Data
    df = generate_mock_oil_data(days=500)
    
    # 2. Setup Engine
    engine = PricePredictorEngine(sequence_length=30)
    X, y = engine.prepare_data(df)
    
    # 3. Train Model
    engine.train(X, y, epochs=50)
    
    # 4. Predict next 30 days
    last_30_days = X[-1]
    future_prices = engine.predict_future(last_30_days, days_to_predict=30)
    
    print("Predicted Oil Prices for the next 5 days:")
    for i, price in enumerate(future_prices[:5]):
        print(f"Day {i+1}: ${price:.2f} per barrel")
