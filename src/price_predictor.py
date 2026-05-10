import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime
import yfinance as yf

# -------------------------------------------------------------
# 1. REAL DATA FETCHER - WTI Crude Oil from Yahoo Finance
#    Ticker "CL=F" = WTI Crude Oil Futures (the global benchmark price)
#    This fetches REAL daily closing prices going back 5 years.
#    Falls back to synthetic data if internet is unavailable.
# -------------------------------------------------------------
def generate_mock_oil_data(days=500):
    """
    Fetches real WTI Crude Oil price data from Yahoo Finance.
    The function name is kept as-is so dashboard.py requires no changes.
    'days' parameter is ignored — we always fetch 5 years of real data.
    """
    print("Fetching real WTI Crude Oil prices from Yahoo Finance...")
    try:
        # CL=F is the Yahoo Finance ticker for WTI Crude Oil Futures
        ticker = yf.Ticker("CL=F")
        df_raw = ticker.history(period="5y")  # 5 years of real daily data

        if df_raw.empty:
            raise ValueError("Yahoo Finance returned empty data.")

        # Keep only the closing price and rename to 'Price'
        df = df_raw[['Close']].rename(columns={'Close': 'Price'})
        df.index = pd.to_datetime(df.index).tz_localize(None)  # remove timezone

        print(f"Successfully loaded {len(df)} days of real WTI crude oil data.")
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Latest price: ${df['Price'].iloc[-1]:.2f} per barrel\n")
        return df

    except Exception as e:
        # Graceful fallback: if no internet, use synthetic data
        print(f"Could not fetch real data ({e}). Falling back to synthetic data...")
        np.random.seed(42)
        base_price = 75.0
        prices = [base_price]
        for _ in range(days - 1):
            volatility = np.random.normal(loc=0.01, scale=1.5)
            new_price = prices[-1] + volatility
            prices.append(max(20.0, new_price))
        dates = pd.date_range(end=datetime.date.today(), periods=days)
        df = pd.DataFrame({'Date': dates, 'Price': prices})
        df.set_index('Date', inplace=True)
        return df

# -------------------------------------------------------------
# 2. LSTM MODEL DEFINITION — DIRECT MULTI-OUTPUT
#
# KEY FIX: Old model predicted ONE day and fed that prediction
# back as input to predict the next day (autoregressive). This
# caused errors to pile up, creating the unrealistic linear slope.
#
# New model predicts ALL output_days at once in a single forward
# pass — no error accumulation, much more realistic predictions.
# Architecture is also stronger: 2 LSTM layers, 128 hidden units,
# and dropout for regularization.
# -------------------------------------------------------------
class OilPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 output_days=30, dropout=0.2):
        super(OilPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_days = output_days

        # 2-layer LSTM with dropout between layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)

        # Single FC layer maps LSTM output → all future days at once
        self.fc = nn.Linear(hidden_size, output_days)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        # Use only the last timestep's hidden state to forecast all future days
        out = self.fc(out[:, -1, :])   # shape: (batch, output_days)
        return out

# -------------------------------------------------------------
# 3. TRAINING & PREDICTION ENGINE
# -------------------------------------------------------------
class PricePredictorEngine:
    def __init__(self, sequence_length=30, output_days=30):
        self.seq_length   = sequence_length
        self.output_days  = output_days
        self.scaler       = MinMaxScaler(feature_range=(-1, 1))
        self.model        = OilPriceLSTM(output_days=output_days)
        self.criterion    = nn.MSELoss()
        self.optimizer    = torch.optim.Adam(self.model.parameters(), lr=0.005)

    def prepare_data(self, df):
        price_data      = df['Price'].values.reshape(-1, 1)
        normalized_data = self.scaler.fit_transform(price_data)

        X, y = [], []
        # Each sample: past seq_length days → next output_days days (all at once)
        for i in range(len(normalized_data) - self.seq_length - self.output_days + 1):
            X.append(normalized_data[i : i + self.seq_length])
            y.append(normalized_data[i + self.seq_length :
                                     i + self.seq_length + self.output_days].flatten())

        return (torch.FloatTensor(np.array(X)),
                torch.FloatTensor(np.array(y)))   # y shape: (N, output_days)

    def train(self, X, y, epochs=50):
        print(f"Training LSTM Model for {epochs} epochs...")
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X)           # (N, output_days)
            loss    = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
        print("Training Complete.\n")

    def predict_future(self, last_sequence, days_to_predict=30):
        """Single forward pass → all future days predicted simultaneously."""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(last_sequence.unsqueeze(0))  # (1, output_days)
        predicted_prices = self.scaler.inverse_transform(
            preds.numpy().reshape(-1, 1))
        return predicted_prices.flatten()[:days_to_predict]

# -------------------------------------------------------------
# EXECUTION (Demonstration)
# -------------------------------------------------------------
if __name__ == "__main__":
    df = generate_mock_oil_data()

    engine   = PricePredictorEngine(sequence_length=30, output_days=30)
    X, y     = engine.prepare_data(df)
    engine.train(X, y, epochs=50)

    last_30_days  = X[-1]
    future_prices = engine.predict_future(last_30_days, days_to_predict=30)

    print("Predicted Oil Prices for the next 5 days:")
    for i, price in enumerate(future_prices[:5]):
        print(f"Day {i+1}: ${price:.2f} per barrel")
