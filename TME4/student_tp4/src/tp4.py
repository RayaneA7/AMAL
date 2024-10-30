import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
from textloader import *
from generate import *

#  TODO:

DATA_PATH = "../../KEBIR_AZIZI_TP3_AMAL/TP3_AMAL/data/trump_full_speech.txt"


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


def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    Calcule la perte de cross-entropy en ignorant les caractères de padding.

    :param output: Tenseur de dimensions (length x batch x output_dim),
                   les logits de la sortie du modèle.
    :param target: Tenseur de dimensions (length x batch), les labels cibles.
    :param padcar: Index du caractère de padding.

    :return: La perte scalaire moyennée sur les éléments non nuls.
    """
    # Calcul de la perte pour chaque élément avec reduction='none'
    loss = F.cross_entropy(
        output.view(
            -1, output.size(-1)
        ),  # Mettre output en (N, C) pour calculer la CE.
        target.view(-1),  # Mettre target en (N,)
        reduction="none",
    )

    # Reshape la perte pour correspondre aux dimensions (length x batch)
    loss = loss.view(target.shape)

    # Créer un masque binaire (1 pour les positions à garder, 0 pour les paddings)
    mask = (target != padcar).float()

    # Appliquer le masque sur la perte
    loss = loss * mask

    # Calculer la moyenne des éléments non nuls (somme divisée par la somme des valeurs du masque)
    return loss.sum() / mask.sum()


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Embedding layer
        self.linear = nn.Linear(embed_size + hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()  # Common choice for RNNs

    def one_step(self, x, h):
        # print(f"{x.shape=}")
        # print(f"{h.shape=}")
        combined = torch.cat((x, h), dim=1)  # Concatenate along feature dimension
        # print(f"{combined.shape=}")
        h_next = self.activation(self.linear(combined))  # Apply non-linearity
        return h_next

    def forward(self, x, h):
        outputs = []
        # Iterate over each time step
        for t in range(x.size(1)):  # x.size(1) is the sequence length
            xi = x[:, t]
            # print(f"{xi.shape=}")
            # Get input at time step t for all batches, shape (batch_size,)
            embedded_x = self.embedding(xi)  # Shape (batch_size, embed_size)
            # print(f"{embedded_x.shape=}")
            h = self.one_step(embedded_x, h)  # Get next hidden state
            output = self.decoder(h)  # Decode hidden state into vocabulary space
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)  # Stack outputs along the time dimension
        return outputs, h  # Return all outputs and the final hidden state

    def decode(self, h):
        return self.decoder(h)


class LSTM(RNN):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super().__init__(vocab_size, embed_size, hidden_size, output_size)

        # Define gates as linear layers
        self.forget_gate = nn.Linear(embed_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(embed_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(embed_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(embed_size + hidden_size, hidden_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.Tanh()

    def one_step(self, x, h, c):
        # Combine the input and hidden state
        combined = torch.cat((x, h), dim=1)  # Concatenate along feature dimension

        # Compute the LSTM gates
        forget = self.sigmoid(self.forget_gate(combined))  # Forget gate
        input_gate = self.sigmoid(self.input_gate(combined))  # Input gate
        output_gate = self.sigmoid(self.output_gate(combined))  # Output gate
        cell_input = self.activation(self.cell_gate(combined))  # Candidate cell state

        # Update the cell state
        c_next = forget * c + input_gate * cell_input

        # Compute the next hidden state
        h_next = output_gate * self.activation(c_next)

        return h_next, c_next

    def forward(self, x, h, c):
        outputs = []

        # Iterate over each time step in the input sequence
        for t in range(x.size(1)):  # x.size(1) is the sequence length
            xi = x[
                :, t
            ]  # Get input at time step t for all batches, shape (batch_size,)
            embedded_x = self.embedding(xi)  # Shape (batch_size, embed_size)
            # Compute next hidden and cell states
            h, c = self.one_step(embedded_x, h, c)

            # Decode hidden state into vocabulary space
            output = self.decode(h)
            outputs.append(output)

        # Stack outputs along the time dimension to get the sequence of outputs
        outputs = torch.stack(
            outputs, dim=1
        )  # Shape: (batch_size, sequence_length, vocab_size)

        return outputs, (
            h,
            c,
        )  # Return all outputs and the final hidden and cell states

    def decode(self, h):
        # Decode hidden state into vocabulary space
        return self.decoder(h)


class GRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Embedding layer
        self.update_gate = nn.Linear(embed_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(embed_size + hidden_size, hidden_size)
        self.candidate_gate = nn.Linear(embed_size + hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.Tanh()

    def one_step(self, x, h):
        combined = torch.cat((x, h), dim=1)  # Concatenate input and hidden state

        # Compute gates
        update = self.sigmoid(self.update_gate(combined))  # Update gate
        reset = self.sigmoid(self.reset_gate(combined))  # Reset gate

        # Compute candidate hidden state
        combined_reset = torch.cat((x, reset * h), dim=1)
        h_tilde = self.activation(self.candidate_gate(combined_reset))

        # Compute the next hidden state
        h_next = update * h + (1 - update) * h_tilde

        return h_next

    def forward(self, x, h):
        outputs = []
        # Iterate over each time step
        for t in range(x.size(1)):  # x.size(1) is the sequence length
            xi = x[
                :, t
            ]  # Get input at time step t for all batches, shape (batch_size,)
            embedded_x = self.embedding(xi)  # Shape (batch_size, embed_size)
            h = self.one_step(embedded_x, h)  # Get next hidden state
            output = self.decoder(h)  # Decode hidden state into vocabulary space
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)  # Stack outputs along the time dimension
        return outputs, h  # Return all outputs and the final hidden state

    def decode(self, h):
        return self.decoder(h)


# class LSTM(RNN):
#     #  TODO:  Implémenter un LSTM


# class GRU(nn.Module):
#     #  TODO:  Implémenter un GRU


# #  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot

# Initialize the model


vocab_size = len(lettre2id) + 1  # Including the NULL character
embed_size = 64  # Size of embedding vector
hidden_size = 128  # Size of hidden state
output_size = vocab_size  # Size of output (vocab size)


# model = LSTM(vocab_size, embed_size, hidden_size, output_size)
model = RNN(vocab_size, embed_size, hidden_size, output_size)


# Hyperparameters
batch_size = 32
epochs = 5
learning_rate = 0.001

# Loss and optimizer
# criterion = maskedCrossEntropy()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

ds = TextDataset(open(DATA_PATH, "rb").read().decode())
# print(f"{ds.phrases=}")
data_trump = DataLoader(
    ds,
    collate_fn=pad_collate_fn,
    batch_size=batch_size,
    shuffle=True,
)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, input_seq in tqdm.tqdm(list(enumerate(data_trump))):

        # Move data to the appropriate device
        # input_seq = input_seq.to(device)
        # target_seq = target_seq.to(device)

        # print(f"{input_seq.shape=}")

        # Initial hidden state (can be zeros or learned)
        input_seq = input_seq.to(torch.long)
        h = torch.zeros(input_seq.size(0), hidden_size)
        # h = torch.zeros(input_seq.size(0), batch_size, hidden_size)
        c = torch.zeros(input_seq.size(0), hidden_size)

        # Forward pass through the RNN
        # output, _ = model(input_seq, h, c)
        output, _ = model(input_seq, h)

        # Reshape the output and target to be batch_size * seq_length x vocab_size
        output = output.view(-1, vocab_size)
        output_seq = input_seq.reshape(-1)  # Use reshape instead of view

        # output_seq = input_seq.view(-1)

        # Compute loss
        loss = maskedCrossEntropy(output, output_seq, padcar=PAD_IX)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(data_trump)}")

print("Training completed.")

model.eval()


print("Generating sequence...")

# generated_seq = generate(
#     model,
#     model.embedding,
#     model.decode,
#     EOS_IX,
#     initial_hidden,
#     maxlen=100,  # Use maxlen for sequence length
# )

# generated_seq = generate_greedy(model, initial_input, initial_hidden, 100)

# generated_seq = generate_beam(
#     model,
#     model.embedding,
#     model.decode,
#     EOS_IX,
#     initial_hidden,
#     maxlen=100,  # Use maxlen for sequence length
# )


# generated_seq = generate(model, model.embedding, model.decode, EOS_IX, maxlen=100)
# generated_seq = generate(
#     model, model.embedding, model.decode, EOS_IX, start="Hello", maxlen=100
# )

start_string = "F"
num_generate = 50  # Number of characters to generate
generated_text = generate_n(model, EOS_IX, start_string, num_generate)
print(generated_text)
# generated_seq = generate_n(model, EOS_IX, start="Hello", maxlen=100)

# print(code2string(generated_seq))
