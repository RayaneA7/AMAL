from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

HIDDEN_SIZE = 256
OUTPUT_SIZE = 2
EPOCHS = 100
LEARNING_RATE = 0.001

PATH = "../data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#  TODO:  Question 3 : Prédiction de séries temporelles
model = RNN(DIM_INPUT*CLASSES, HIDDEN_SIZE, OUTPUT_SIZE*CLASSES).to(device)
criterion = torch.nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train(model, data_train, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in data_train:
            sequences, targets = batch  # sequences: (batch_size, length, dim), targets: (batch_size, length, 2)
            sequences = sequences.view(sequences.size(0), sequences.size(1), -1)
            sequences = sequences.permute(1,0,2)
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Initialize the hidden state (batch_size, hidden_size)
            h = torch.zeros(sequences.size(1), HIDDEN_SIZE).to(device)

            # Forward pass through RNN
            hidden_states = model(sequences, h)
            predictions = model.decode(
                hidden_states.view(hidden_states.size(0)*hidden_states.size(1), hidden_states.size(2)), 
            )
            predictions = predictions.view(targets.size(1),targets.size(0),targets.size(2),targets.size(3))
            predictions = predictions.permute(1,0,2,3)

            # Compute loss
            loss = criterion(predictions, targets)
            total_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_train)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Testing Loop
def test(model, data_test, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_test:
            sequences, targets = batch
            sequences = sequences.view(sequences.size(0), sequences.size(1), -1)
            sequences = sequences.permute(1,0,2)
            sequences, targets = sequences.to(device), targets.to(device)

            # Initialize hidden state
            h = torch.zeros(sequences.size(1), HIDDEN_SIZE).to(device)

            # Forward pass
            hidden_states = model(sequences, h)
            predictions = model.decode(
                hidden_states.view(hidden_states.size(0)*hidden_states.size(1), hidden_states.size(2)), 
            )
            predictions = predictions.view(targets.size(1),targets.size(0),targets.size(2),targets.size(3))
            predictions = predictions.permute(1,0,2,3)

            # Compute loss
            loss = criterion(predictions, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_test)
    print(f"Test Loss: {avg_loss:.4f}")

# Train the model
train(model, data_train, criterion, optimizer, EPOCHS)

# Test the model
test(model, data_test, criterion)
