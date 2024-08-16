import torch
from io import open
import numpy as np
from conllu import parse_incr
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import CreateDataset, LSTMTagger, train_LSTM

# variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cut_off_freq = 3
batch_size = 32

# loading the train data
data_file = open("data/en_atis-ud-train.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(data_file))
data_file.close()

# train data
train_data = CreateDataset(sentences,cut_off_freq=cut_off_freq,method='LSTM')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# loading the validation data
data_file = open("data/en_atis-ud-dev.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(data_file))
data_file.close()

# validation data
val_data = CreateDataset(sentences, cut_off_freq=cut_off_freq, vocab=train_data.vocab, upos_vocab=train_data.upos_vocab,method='LSTM')
val_loader = DataLoader(val_data, batch_size=batch_size)

# Hyperparameters for the model
embedding_dim = 128
hidden_dim = 128
output_dim = len(train_data.upos_vocab)
input_dim = len(train_data.vocab)
n_layers = 1
lr = 0.001
bidirectionality = True
activation = "relu"
epochs = 15

# Model
model = LSTMTagger(input_dim, embedding_dim, hidden_dim, output_dim,n_layers,activation,bidirectionality).to(device)

loss_train, loss_val, model = train_LSTM(model, train_loader, val_loader, lr, device, epochs)

# Plotting the loss
plt.plot(loss_train, label="train")
plt.plot(loss_val, label="val")
plt.legend()
plt.show()

# Saving model
torch.save(model, "lstm_model.pth")

# saving Vocabularies
np.save("vocab_lstm.npy", train_data.vocab)
np.save("upos_vocab_lstm.npy", train_data.upos_vocab)
print("Training Done!")
