import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. Download data ---
ticker = "AAPL"
raw_data = yf.download(ticker, start="2015-01-01", end="2023-12-31")
data = raw_data.copy()

# --- 2. Flatten MultiIndex if needed ---
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

print("Flattened columns:", data.columns)

# --- 3. Extract 'Close' column safely ---
close_col = f"{ticker}_Close"
if close_col not in data.columns:
    raise ValueError(f"Column '{close_col}' not found in flattened data.")

prices = data[close_col]
returns = prices.pct_change().dropna()

# --- 4. Create sliding window features ---
window_size = 10
X, y = [], []

for i in range(window_size, len(returns)):
    X.append(returns[i - window_size:i].values)
    y.append(returns[i])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

# --- 5. Define a basic neural network ---
model = nn.Sequential(
    nn.Linear(window_size, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 6. Train the model ---
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: Loss = {loss.item():.6f}")
