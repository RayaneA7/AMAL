import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import tqdm

from utils import RNN, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation + string.digits + " "
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1, len(LETTRES) + 1), LETTRES))
id2lettre[0] = ""  ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))

PATH = "../data/"


def normalize(s):
    """Nettoyage d'une chaîne de caractères."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if c in LETTRES)


def string2code(s):
    """Transformation d'une chaîne de caractère en tenseur d'indexes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])


def code2string(t):
    """Transformation d'une liste d'indexes en chaîne de caractères"""
    seq = ""
    for i in t:
        seq += " " + "".join([id2lettre[j] for j in i])
    return seq


class TrumpDataset(Dataset):
    def __init__(self, text, maxsent=None, maxlen=None):
        """Dataset pour les tweets de Trump
        * text : texte brut
        * maxsent : nombre maximum de phrases.
        * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [
            p[:maxlen].strip() + "." for p in full_text.split(".") if len(p) > 0
        ]
        if maxsent is not None:
            self.phrases = self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN - t.size(0), dtype=torch.long), t])
        return t[:-1], t[1:]


import torch
import torch.nn as nn
import torch.nn.functional as F


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Embedding layer
        self.hidden_size = hidden_size
        self.rnn_cell = nn.Linear(embed_size + hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()

    def one_step(self, x, h):
        combined = torch.cat(
            (x, h), dim=1
        )  # Concatenate embedded input and hidden state
        h_next = self.activation(self.rnn_cell(combined))
        return h_next

    def forward(self, x, h):
        outputs = []
        h_next = h
        for t in range(x.size(1)):  # Iterate over sequence length
            embedded = self.embedding(x[:, t])  # Embed the input (character)
            h_next = self.one_step(embedded, h_next)
            output = self.decoder(h_next)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # Stack along the sequence dimension
        return outputs, h_next

    def generate(self, initial_input, initial_hidden, seq_length):
        """Generate a sequence of characters from an initial input."""
        generated_sequence = []
        input = initial_input
        hidden = initial_hidden

        for _ in range(seq_length):
            embedded = self.embedding(input)
            hidden = self.one_step(embedded, hidden)
            output = self.decoder(hidden)
            output_prob = F.softmax(
                output, dim=1
            )  # Convert to probability distribution
            input = torch.argmax(
                output_prob, dim=1
            )  # Choose the most probable next character
            generated_sequence.append(input)

        return torch.stack(generated_sequence, dim=1)


# Parameters for the model
vocab_size = len(LETTRES) + 1  # Including the NULL character
embed_size = 32  # Size of embedding vector
hidden_size = 128  # Hidden state size
output_size = vocab_size  # Output size equals number of symbols in vocabulary

# Initialize the model
model = CharRNN(vocab_size, embed_size, hidden_size, output_size)


# Hyperparameters
batch_size = 128
epochs = 20
learning_rate = 0.01

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

data_trump = DataLoader(
    TrumpDataset(
        open(PATH + "trump_full_speech.txt", "rb").read().decode(), maxlen=1000
    ),
    batch_size=batch_size,
    shuffle=True,
)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (input_seq, target_seq) in tqdm.tqdm(list(enumerate(data_trump))):
        optimizer.zero_grad()

        # Move data to the appropriate device
        # input_seq = input_seq.to(device)
        # target_seq = target_seq.to(device)

        # print(f"{input_seq.shape=}")

        # Initial hidden state (can be zeros or learned)
        h = torch.zeros(input_seq.size(0), hidden_size)

        # Forward pass through the RNN
        output, _ = model(input_seq, h)

        # Reshape the output and target to be batch_size * seq_length x vocab_size
        output = output.view(-1, vocab_size)
        target_seq = target_seq.view(-1)

        # Compute loss
        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(data_trump)}")

print("Training completed.")

model.eval()
initial_input = torch.tensor([lettre2id["T"]])  # Start the sequence with 'T'
initial_hidden = torch.zeros(1, hidden_size)  # Initialize hidden state
generated_seq = model.generate(initial_input, initial_hidden, seq_length=100)
print(code2string(generated_seq.tolist()))
