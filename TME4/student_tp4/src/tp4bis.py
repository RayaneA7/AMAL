import torch
from torch.utils.data import Dataset
from torch import device
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import tqdm
from textloader import *
from generate import *
from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weight matrices for input to hidden and hidden to hidden transitions
        self.W_ih = nn.Linear(embedding_dim, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, output_size)

    def one_step(self, x, hidden):
        """
        Process one time step.
        x: Input tensor of shape (batch_size, embedding_dim)
        hidden: Hidden state tensor of shape (batch_size, hidden_size)
        """
        hidden = torch.tanh(self.W_ih(x) + self.W_hh(hidden))
        return hidden

    def decode(self, hidden_states):
        """
        Decode the hidden states into outputs.
        hidden_states: Tensor of shape (seq_len, batch_size, hidden_size)
        Returns:
            decoded_outputs: Tensor of shape (seq_len, batch_size, output_size)
        """
        decoded_outputs = self.W_ho(hidden_states)
        return decoded_outputs

    def forward(self, x_seq, hidden=None):
        """
        Forward pass through the RNN.
        x_seq: Input sequence tensor of shape (seq_len, batch_size)
        hidden: Initial hidden state of shape (batch_size, hidden_size) or None
        Returns:
            hidden_states: Tensor of shape (seq_len, batch_size, hidden_size)
        """
        seq_len, batch_size = x_seq.size()

        # Embed the input sequence
        inputs = x_seq.long()
        embedded = self.embedding(inputs)  # Shape: (seq_len, batch_size, embedding_dim)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=embedded.device)

        hidden_states = []
        for t in range(seq_len):
            x_t = embedded[t]  # Get embedded input at time step t
            hidden = self.one_step(x_t, hidden)  # Update hidden state
            hidden_states.append(hidden)

        hidden_states = torch.stack(hidden_states, dim=0)
        return hidden_states


# Define parameters
vocab_size = (
    len(LETTRES) + 2
)  # Number of input symbols (including the null character and eos)
embedding_dim = 64  # Size of each embedding vector
hidden_size = 128  # Size of the hidden state
output_size = vocab_size  # Number of output symbols (same as input_size)
num_epochs = 10
learning_rate = 0.001
pad_car = 0  # Assuming 0 is the padding character index


# Hyperparameters
batch_size = 32
epochs = 5

# Loss and optimizer
# criterion = maskedCrossEntropy()
ds = TextDataset(open(DATA_PATH, "rb").read().decode())
# print(f"{ds.phrases=}")
data_trump = DataLoader(
    ds,
    collate_fn=pad_collate_fn,
    batch_size=batch_size,
    shuffle=True,
)

# Instantiate the models and move them to the device
rnn_model = RNN(vocab_size, embedding_dim, hidden_size, output_size)

# Loss function and optimizer
optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)

# TensorBoard setup
writer = SummaryWriter(log_dir="runs/trump_rnn")

# Training loop
for epoch in range(num_epochs):
    rnn_model.train()
    total_loss = 0

    for i, inputs in enumerate(data_trump):
        # Move inputs and targets to the device
        targets = inputs.clone()
        targets = inputs[1:, :]
        inputs = inputs[:-1, :]

        inputs, targets = inputs, targets

        hidden_state = torch.zeros(inputs.size(1), hidden_size)

        # Pass through the RNN model
        hidden_states = rnn_model(
            inputs, hidden_state
        )  # (seq_len, batch_size, hidden_size)

        outputs = rnn_model.decode(hidden_states)  # (seq_len, batch_size, output_size)

        # Reshape outputs and targets for computing loss
        outputs = outputs.reshape(
            -1, output_size
        )  # (seq_len * batch_size, output_size)
        targets = targets.reshape(-1).long()  # (seq_len * batch_size)

        # Compute the loss using masked cross-entropy
        loss = maskedCrossEntropy(outputs, targets, pad_car)
        total_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss to TensorBoard every 100 batches
        if (i + 1) % 100 == 0:
            writer.add_scalar("Loss/train", loss.item(), epoch * len(data_trump) + i)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(data_trump)}], Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(data_trump)
    writer.add_scalar("Loss/epoch", avg_loss, epoch + 1)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Close the TensorBoard writer
writer.close()
