from utils import RNN, device,SampleMetroDataset
import torch
from torch.utils.data import DataLoader

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

HIDDEN_SIZE = 128
OUTPUT_SIZE = CLASSES
EPOCHS = 100
LEARNING_RATE = 0.001

PATH = "../data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)


#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
model = RNN(DIM_INPUT, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
criterion = torch.nn.CrossEntropyLoss()  # Multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train(model, data_train, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in data_train:
            sequences, labels = batch  # sequences: (batch, length, dim), labels: (batch)
            sequences = sequences.permute(1,0,2) #(length, batch, dim)
            sequences, labels = sequences.to(device), labels.to(device)

            # print(f"{sequences.shape=}")
            
            # Initial hidden state
            h = torch.zeros(sequences.size(1), HIDDEN_SIZE).to(device)

            # Forward pass through RNN
            hidden_states = model(sequences, h)
            
            # Decode the last hidden state
            outputs = model.decode(hidden_states[-1])

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(data_train)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Testing Loop
def test(model, data_test, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_test:
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)
            sequences = sequences.permute(1,0,2) #(length, batch, dim)

            # Initial hidden state
            h = torch.zeros(sequences.size(1), HIDDEN_SIZE).to(device)

            # Forward pass
            hidden_states = model(sequences, h)

            # Decode the last hidden state
            outputs = model.decode(hidden_states[-1])

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_test)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

# Train the model
train(model, data_train, criterion, optimizer, EPOCHS)

# Test the model
test(model, data_test, criterion)
