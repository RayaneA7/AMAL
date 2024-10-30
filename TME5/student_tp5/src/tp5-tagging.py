import itertools
import logging
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
from conllu import parse_incr

logging.basicConfig(level=logging.INFO)

DATA_PATH = "data/"


# Format de sortie décrit dans
# https://pypi.org/project/conllu/


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """

    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """oov : autorise ou non les mots OOV"""
        self.oov = oov
        self.id2word = ["PAD"]
        self.word2id = {"PAD": Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]


class TaggingDataset:
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(
                (
                    [words.get(token["form"], adding) for token in s],
                    [tags.get(token["upostag"], adding) for token in s],
                )
            )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(
        pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2)
    )


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)

data_file = open(DATA_PATH + "fr_gsd-ud-train.conllu")
# raw_train = [parse(x)[0] for x in data_file if len(x) > 1]<
train_data = TaggingDataset(parse_incr(data_file), words, tags, True)
data_file = open(DATA_PATH + "fr_gsd-ud-dev.conllu")
dev_data = TaggingDataset(parse_incr(data_file), words, tags, True)
# dev_data = TaggingDataset(raw_dev, words, tags, True)

# raw_dev = TaggingDataset(parse_incr(data_file), words, tags, True)
data_file = open(DATA_PATH + "fr_gsd-ud-test.conllu")
# raw_test = [parse(x)[0] for x in data_file if len(x) > 1]
# test_data = TaggingDataset(raw_test, words, tags, False)
test_data = TaggingDataset(parse_incr(data_file), words, tags, False)

# train_data = TaggingDataset(raw_train, words, tags, True)


logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE = 100

train_loader = DataLoader(
    train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True
)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)


#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)


class Seq2SeqTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, padding_idx):
        super(Seq2SeqTagger, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        tag_scores = self.fc(lstm_out)
        return tag_scores


def train_model(
    model, train_loader, dev_loader, criterion, optimizer, vocab, num_epochs=10
):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(train_loader):
            # Remplacement aléatoire des mots par OOV pendant l'apprentissage
            inputs = torch.where(
                torch.rand(inputs.size()) < 0.1,
                torch.full_like(inputs, vocab["__OOV__"]),
                inputs,
            )

            # Calcul des longueurs des séquences (sans padding)
            lengths = [len(seq[seq != 0]) for seq in inputs]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs, lengths)

            # Reshape outputs and targets for loss calculation (ignoring padding)
            outputs = outputs.view(
                -1, outputs.shape[-1]
            )  # [batch_size * seq_len, num_classes]
            targets = targets.view(-1)  # [batch_size * seq_len]

            # Calcul de la loss (on ignore le padding)
            # print(outputs.shape, targets.shape)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

        # Évaluation sur le jeu de validation
        evaluate_model(model, dev_loader)


def evaluate_model(model, dev_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in dev_loader:
            lengths = [len(seq[seq != 0]) for seq in inputs]
            outputs = model(inputs, lengths)
            predictions = outputs.argmax(dim=-1)

            # Calcul de la précision (en ignorant les indices de padding)
            mask = targets != Vocabulary.PAD
            correct += (predictions == targets)[mask].sum().item()
            total += mask.sum().item()

    print(f"Accuracy: {correct / total:.4f}")


def predict_sentence_tags(model, sentence, vocab_words, vocab_tags):
    model.eval()
    inputs = torch.LongTensor([vocab_words[word] for word in sentence])
    lengths = [len(inputs)]
    inputs = inputs.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(inputs, lengths)
        predictions = outputs.argmax(dim=-1)

    return vocab_tags.getwords(predictions[0].tolist())


def main():
    # Paramètres du modèle
    vocab_size = len(words)  # Taille du vocabulaire
    tagset_size = len(tags)  # Nombre de tags possibles
    embedding_dim = 100  # Dimension des embeddings
    hidden_dim = 128  # Dimension cachée du LSTM
    padding_idx = Vocabulary.PAD  # Index de padding

    # Initialisation du modèle, de la fonction de coût et de l'optimiseur
    model = Seq2SeqTagger(
        vocab_size, tagset_size, embedding_dim, hidden_dim, padding_idx
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=padding_idx
    )  # On ignore le padding dans la perte
    optimizer = optim.Adam(model.parameters(), lr=0.006)

    # Nombre d'époques d'entraînement
    num_epochs = 10

    # Entraînement du modèle
    print("Training the model...")
    train_model(
        model,
        train_loader,
        dev_loader,
        criterion,
        optimizer,
        vocab=words,
        num_epochs=num_epochs,
    )

    # Évaluation finale sur le jeu de test
    print("Evaluating on the test set...")
    evaluate_model(model, test_loader)

    # Exemple de prédiction sur une phrase test
    sentence = ["Je", "suis", "un", "étudiant", "ESI"]
    print(f"Sentence: {sentence}")
    predicted_tags = predict_sentence_tags(model, sentence, words, tags)
    print(f"Predicted tags: {predicted_tags}")


if __name__ == "__main__":
    main()
