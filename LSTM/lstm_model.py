# LSTM Sales Prediction
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv"
dataframe = pd.read_csv(url, usecols=[1], skipfooter=2, engine='python')
dataset = dataframe.values.astype('float32')

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Convert dataset into sequences and corresponding labels
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :])
        return out

# Set random seed for reproducibility
torch.manual_seed(0)

# Set hyperparameters
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
learning_rate = 0.01
num_epochs = 100

# Create model, loss function, and optimizer
model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    inputs = torch.tensor(trainX, dtype=torch.float)
    labels = torch.tensor(trainY, dtype=torch.float)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch} loss: {loss.item()}')

# Predict on test data
test_inputs = torch.tensor(testX, dtype=torch.float)
predicted = model(test_inputs).detach().numpy()
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))

# Predict using the test dataset

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(testY, predicted))
print(f"Test RMSE: {rmse}")

# Plot the results
actual = scaler.inverse_transform([testY])

plt.figure(figsize=(15, 6))
plt.plot(actual[0], label="Actual Sales")
plt.plot(predicted, label="Predicted Sales")
plt.legend()
plt.title("Sales Prediction using LSTM")
plt.show()
