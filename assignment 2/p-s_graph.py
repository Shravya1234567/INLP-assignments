import torch
import torch.nn as nn
from io import open
import numpy as np
from conllu import parse_incr
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import CreateDataset, FFNN

# variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
p = [1,2,3,4]

# loading vocabularies
vocab = np.load("vocab.npy", allow_pickle=True).item()
upos_vocab = np.load("upos_vocab.npy", allow_pickle=True).item()

loss1 = []
acc = []

for i in p:
    # loading the train data
    data_file = open("data/en_atis-ud-train.conllu", "r", encoding="utf-8")
    sentences = list(parse_incr(data_file))
    data_file.close()
    # train data
    train_data = CreateDataset(sentences, i, i, 3, upos_vocab, vocab)

    # loading the validation data
    data_file = open("data/en_atis-ud-dev.conllu", "r", encoding="utf-8")
    sentences = list(parse_incr(data_file))
    data_file.close()

    # validation data
    val_data = CreateDataset(sentences, i, i, 3, upos_vocab, vocab)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Hyperparameters for the model
    emmbedding_dim = 256
    hidden_dim = 64
    output_dim = len(upos_vocab)
    lr = 0.001
    n_layers = 1
    activation = "tanh"
    epochs = 5

    # Model
    model = FFNN(len(vocab), emmbedding_dim, hidden_dim, output_dim, i, i, vocab, n_layers,activation).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_val = []
    loss_train = []
    accuracy_val = []
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        for j, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            avg_loss += loss.item()
            if j % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{j + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        loss_train.append(avg_loss / len(train_loader))
        print(f"EPOCH {epoch + 1} LOSS: {avg_loss / len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            crct = 0
            total = 0
            for j, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                total += labels.size(0)
                crct += (outputs.argmax(1) == labels.argmax(1)).sum().item()
            print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
            loss_val.append(val_loss / len(val_loader))  
            accuracy_val.append(crct / total)  

    loss1.append(loss_val[-1])
    acc.append(accuracy_val[-1])

plt.plot(p, loss1)
plt.xlabel("p")
plt.ylabel("Loss")
plt.title("Loss vs p")
plt.show()

plt.plot(p, acc)
plt.xlabel("p")
plt.ylabel("Accuracy")
plt.title("Accuracy vs p")
plt.show()
