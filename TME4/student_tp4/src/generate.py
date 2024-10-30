from textloader import string2code, id2lettre
import math
import torch
from torch.nn import functional as F
from textloader import lettre2id

#  TODO:  Ce fichier contient les différentes fonction de génération
hidden_size = 128


def generate_greedy(rnn, initial_input, initial_hidden, seq_length):
    """Generate a sequence of characters from an initial input."""
    generated_sequence = []
    input = initial_input
    hidden = initial_hidden

    for _ in range(seq_length):
        embedded = rnn.embedding(input)
        hidden = rnn.one_step(embedded, hidden)
        output = rnn.decoder(hidden)
        output_prob = F.softmax(output, dim=1)  # Convert to probability distribution
        input = torch.argmax(
            output_prob, dim=1
        )  # Choose the most probable next character
        generated_sequence.append(input)

    return torch.stack(generated_sequence, dim=0)


def generate_n(rnn, eos, start="", maxlen=200):
    """
    Fonction de génération (l'embedding et le décodeur sont des fonctions du rnn).
    Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200
    ou qui s'arrête quand eos est généré.

    Args:
        rnn: Le réseau récurrent (RNN).
        emb: La couche d'embedding.
        decoder: Le décodeur pour transformer l'état caché en logits.
        eos: ID du token end of sequence (EOS).
        start: Début de la phrase.
        maxlen: Longueur maximale de la séquence générée.

    Returns:
        generated_text (str): La séquence générée.
    """
    # Préparation de l'input initial
    device = next(rnn.parameters()).device  # Assure que tout est sur le même device
    if start:
        input_eval = string2code(start).unsqueeze(0).to(device)  # Shape (1, seq_len)
        input_eval = input_eval.permute(1, 0)  # Shape (seq_len, batch_size=1)
    else:
        input_eval = torch.tensor(
            [[0]], device=device
        )  # Si start est vide, on initialise à 0

    hidden_state_initial = torch.zeros(
        1, rnn.hidden_size, device=device
    )  # État caché initialisé à zéro
    generated_text = start

    for i in range(maxlen):

        # Passage à travers le RNN

        print(f"{input_eval.shape=}")
        print(f"{hidden_state_initial.shape=}")
        _, hidden_state = rnn(
            input_eval, hidden_state_initial
        )  # Output de forme (seq_len, batch_size, hidden_size)

        output = rnn.decode(hidden_state)
        print(output[-1].shape)

        # On décode l'état caché pour obtenir les logits
        logits = output[-1]  # On prend le dernier état caché

        # Choix du caractère le plus probable
        predicted_id = torch.argmax(logits, dim=-1).item()

        # Ajouter le caractère prédicté à la séquence générée
        generated_text += id2lettre[predicted_id]

        # Vérifier si le token EOS est généré
        if predicted_id == eos:
            break

        # Prepare the next input
        predicted_id_tensor = torch.tensor(
            [[predicted_id]], dtype=torch.long, device=device
        )  # Shape (1, 1)

        # Update input_eval to keep its dimensions compatible
        input_eval = torch.cat(
            [input_eval, predicted_id_tensor], dim=0
        )  # Update the input

    return generated_text


def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    """
    Generates a sequence of tokens using the RNN and decoder.

    Args:
        rnn (RNN): The RNN model.
        emb (nn.Embedding): The embedding layer.
        decoder (callable): The decoder function.
        eos (int): The ID of the end-of-sequence token.
        start (str): The initial input to the RNN (starting characters).
        maxlen (int): Maximum length of the generated sequence.

    Returns:
        list: The generated sequence of token IDs.
    """
    # Initialize the hidden state
    hidden = torch.zeros(1, hidden_size)

    # Encode the initial input (starting string)
    if start:
        input_seq = string2code(start)  # Convert starting string to code (token IDs)
    else:
        input_seq = torch.tensor([lettre2id["R"]])  # Default start token

    input_seq = input_seq.unsqueeze(0)  # Shape (1, seq_len)

    generated_sequence = []

    for _ in range(maxlen):

        # Embed the current input token
        # embedded_input = emb(input_seq)
        # embedded_input = embedded_input.to(torch.long)
        print(f"{input_seq.shape=}")
        # print(f"{embedded_input.shape=}")
        # Forward pass through the RNN
        output, hidden = rnn(input_seq, hidden)

        # Decode the RNN output to get the next token probabilities
        next_token_logits = decoder(hidden)

        # Sample the next token (can also use argmax for greedy decoding)
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()

        # Append the generated token to the sequence
        generated_sequence.append(next_token_id)

        # Stop if the EOS token is generated
        if next_token_id == eos:
            break

        # Prepare the next input (the token just generated)
        input_seq = torch.tensor([[next_token_id]])

    return generated_sequence


def generate_from_lstm(lstm, emb, decoder, eos, start="", maxlen=200):
    """
    Generates a sequence of tokens using the RNN and decoder.

    Args:
        rnn (RNN): The RNN model.
        emb (nn.Embedding): The embedding layer.
        decoder (callable): The decoder function.
        eos (int): The ID of the end-of-sequence token.
        start (str): The initial input to the RNN (starting characters).
        maxlen (int): Maximum length of the generated sequence.

    Returns:
        list: The generated sequence of token IDs.
    """
    # Initialize the hidden state
    hidden = torch.zeros(1, hidden_size)
    c = torch.zeros(1, hidden_size)

    # Encode the initial input (starting string)
    if start:
        input_seq = string2code(start)  # Convert starting string to code (token IDs)
    else:
        input_seq = torch.tensor([lettre2id["R"]])  # Default start token

    input_seq = input_seq.unsqueeze(0)  # Shape (1, seq_len)

    generated_sequence = []

    for _ in range(maxlen):

        # Embed the current input token
        # embedded_input = emb(input_seq)
        # embedded_input = embedded_input.to(torch.long)
        # print(f"{input_seq.shape=}")
        # print(f"{embedded_input.shape=}")
        # Forward pass through the RNN
        output, (hidden, c) = lstm(
            input_seq, hidden, c
        )  # Pass both hidden and cell state
        # Decode the RNN output to get the next token probabilities
        next_token_logits = decoder(hidden)

        # Sample the next token (can also use argmax for greedy decoding)
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()

        # Append the generated token to the sequence
        generated_sequence.append(next_token_id)

        # Stop if the EOS token is generated
        if next_token_id == eos:
            break

        # Prepare the next input (the token just generated)
        input_seq = torch.tensor([[next_token_id]])

    return generated_sequence


def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    # Encodage du début de la phrase
    start_tensor = torch.tensor([start], dtype=torch.long).unsqueeze(0)  # (1, 1)
    start_emb = emb(start_tensor)  # Embedding du token de départ

    # Initialiser le faisceau avec l'état initial
    beam = [
        (start_emb, torch.zeros(rnn.linear.out_features), 0, [start])
    ]  # (input, hidden state, score, generated sequence)
    completed_sequences = []  # Séquences terminées

    # Boucle de génération jusqu'à maxlen
    for _ in range(maxlen):
        candidates = []  # Candidats pour cette itération
        for emb_input, h, score, seq in beam:
            if seq[-1] == eos:  # Si la séquence est terminée, ne plus la prolonger
                completed_sequences.append((seq, score))
                continue

            # Étendre le faisceau avec les nouvelles prédictions
            h_next = rnn.one_step(emb_input, h)  # Calculer l'état suivant
            logits = decoder(h_next)  # Prédire le prochain token
            probs = F.log_softmax(logits, dim=-1)  # Convertir en log-probabilité

            # Obtenir les k meilleurs tokens
            top_k_probs, top_k_ids = torch.topk(probs, k)

            # Ajouter les k nouveaux candidats au faisceau
            for i in range(k):
                new_seq = seq + [top_k_ids[0, i].item()]  # Séquence étendue
                new_score = (
                    score + top_k_probs[0, i].item()
                )  # Ajouter la log-probabilité
                new_emb_input = emb(
                    top_k_ids[0, i].unsqueeze(0)
                )  # Embedding du nouveau token
                candidates.append((new_emb_input, h_next, new_score, new_seq))

        # Trier les candidats par score décroissant et conserver les k meilleurs
        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
        beam = candidates[:k]

        # Si tout le faisceau est composé de séquences terminées, arrêter
        if all(seq[-1] == eos for _, _, _, seq in beam):
            break

    # Ajouter les séquences incomplètes
    completed_sequences.extend(
        (seq, score) for _, _, score, seq in beam if seq[-1] != eos
    )

    # Retourner la meilleure séquence générée
    best_sequence = max(
        completed_sequences, key=lambda x: x[1]
    )  # Séquence avec le meilleur score
    return best_sequence[0]  # Retourner seulement la séquence


def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """

    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        # On décode l'état caché pour obtenir les logits
        logits = decoder(h)
        # On calcule les probabilités avec softmax
        probs = F.softmax(logits, dim=-1)

        # Trier les probabilités et les indices correspondants
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Calculer la somme cumulée des probabilités
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Retenir les tokens dont la somme cumulée est <= alpha
        cutoff_index = torch.searchsorted(cumulative_probs, alpha).item()

        # Garder uniquement les tokens dans le top-p ensemble
        top_p_probs = sorted_probs[: cutoff_index + 1]
        top_p_indices = sorted_indices[: cutoff_index + 1]

        # Normaliser les probabilités sur le sous-ensemble
        top_p_probs /= torch.sum(top_p_probs)

        # Échantillonner selon la distribution restreinte
        next_token = torch.multinomial(top_p_probs, 1).item()

        return top_p_indices[next_token]

    return compute


# p_nucleus
# def p_nucleus(decoder, alpha: float):
#     """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

#     Args:
#         * decoder: renvoie les logits étant donné l'état du RNN
#         * alpha (float): masse de probabilité à couvrir
#     """

#     def compute(h):
#         """Calcule la distribution de probabilité sur les sorties

#         Args:
#            * h (torch.Tensor): L'état à décoder
#         """
#         #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)

#     return compute
