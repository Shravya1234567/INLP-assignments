import torch
from io import open
import numpy as np
from conllu import parse_incr
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import CreateDataset, FFNN, train_FFNN

# variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cut_off_freq = 3
p = 1
s = 1
batch_size = 32

# loading the train data
data_file = open("data/en_atis-ud-train.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(data_file))
data_file.close()

# train data
train_data = CreateDataset(sentences, p, s, cut_off_freq)

# loading the validation data
data_file = open("data/en_atis-ud-dev.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(data_file))
data_file.close()

# validation data
val_data = CreateDataset(sentences, p, s, cut_off_freq, train_data.upos_vocab, train_data.vocab)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# Hyperparameters for the model
emmbedding_dim = 256
hidden_dim = 64
output_dim = len(train_data.upos_vocab)
lr = 0.001
n_layers = 1
activation = "tanh"
epochs = 5

# Model
model = FFNN(len(train_data.vocab), emmbedding_dim, hidden_dim, output_dim, p, s, train_data.vocab,n_layers,activation).to(device)

loss_train, loss_val, model = train_FFNN(model,train_loader,val_loader,epochs,device,lr)

# ploting the loss
plt.plot(loss_train, label="Training Loss")
plt.plot(loss_val, label="Validation Loss")
plt.legend()
plt.show()

# Saving model
torch.save(model, "model.pth")

# saving Vocabularies
np.save("vocab.npy", train_data.vocab)
np.save("upos_vocab.npy", train_data.upos_vocab)
print("Training Done!")