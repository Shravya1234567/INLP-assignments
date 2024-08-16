import torch
import torch.nn as nn
from io import open
import numpy as np
from conllu import parse_incr
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import CreateDataset, LSTMTagger

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
epochs = 20

# Model
model = LSTMTagger(input_dim, embedding_dim, hidden_dim, output_dim,n_layers,activation,bidirectionality).to(device)

model.train()
loss_val = []
accuracy_val = []
loss_train = []
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.shape[-1])
        labels = labels.view(-1, labels.shape[-1])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss_train.append(running_loss/len(train_loader))
    print(f"Epoch: {epoch+1}, Loss: {running_loss/len(train_loader)}")

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_loss = 0
        crct = 0
        total = 0
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1, labels.shape[-1])
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            crct += (predicted == labels).sum().item()
        accuracy_val.append(crct/total)
        loss_val.append(val_loss/len(val_loader))
        print(f"Validation Loss: {val_loss/len(val_loader)}")

# Plotting the graph
plt.plot(range(1,epochs+1),loss_val, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epochs vs Loss")
plt.legend()
plt.show()

plt.plot(range(1,epochs+1),accuracy_val, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Epochs vs Accuracy")
plt.legend()
plt.show()