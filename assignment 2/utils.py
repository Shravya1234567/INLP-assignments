import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class CreateDataset(Dataset):
    def __init__(self, sentences, p=1, s=1, cut_off_freq = 3, upos_vocab=None, vocab=None, method=None):
        self.sentences = self.add_unks(sentences, cut_off_freq, vocab)
        self.p = p
        self.s = s
        self.cut_off_freq = cut_off_freq
        self.method = method
        self.max_len = None
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self.create_vocab()
        if upos_vocab is not None:
            self.upos_vocab = upos_vocab
        else:
            self.upos_vocab = self.create_upos_vocab()
        self.X, self.y = self.create_training_data()

    def add_unks(self, sentences, cut_off_freq, vocab=None):
        if vocab is None:
            counts = {}
            for sentence in sentences:
                for token in sentence:
                    if token["form"] in counts:
                        counts[token["form"]] += 1
                    else:
                        counts[token["form"]] = 1
            for sentence in sentences:
                for token in sentence:
                    if counts[token["form"]] < cut_off_freq:
                        token["form"] = "<unk>"
            return sentences
        else:
            for sentence in sentences:
                for token in sentence:
                    if token["form"] not in vocab:
                        token["form"] = "<unk>"
            return sentences

    def create_vocab(self):
        vocab = set()
        for sentence in self.sentences:
            for token in sentence:
                vocab.add(token["form"])
        vocab.add("<s>")
        vocab.add("</s>")
        vocab.add("<pad>")
        vocab = list(vocab)
        vocab = {token: i for i, token in enumerate(vocab)}
        return vocab
    
    def create_upos_vocab(self):
        upos_vocab = set()
        for sentence in self.sentences:
            for token in sentence:
                upos_vocab.add(token["upos"])
        upos_vocab.add("UNK")
        upos_vocab = list(upos_vocab)
        upos_vocab = {upos: i for i, upos in enumerate(upos_vocab)}
        return upos_vocab
    
    def create_training_data(self):
        X = []
        y = []
        if self.method == 'LSTM':
            max_len = max([len(sentence) for sentence in self.sentences])
            max_len += 2
            self.max_len = max_len
            for sentence in self.sentences:
                sentence.insert(0, {"form": "<s>", "upos": "UNK"})
                sentence.append({"form": "</s>", "upos": "UNK"})
                for i in range(max_len - len(sentence)):
                    sentence.append({"form": "<pad>", "upos": "UNK"})

            for sentence in self.sentences:
                X.append([sentence[j]["form"] for j in range(max_len)])
                y.append([sentence[j]["upos"] for j in range(max_len)])
            X = [[self.vocab[token] for token in x] for x in X]
            y = [[self.upos_vocab[upos] if upos in self.upos_vocab else self.upos_vocab["UNK"] for upos in y] for y in y]
            y = [[np.eye(len(self.upos_vocab))[upos] for upos in y] for y in y]
            return X, y
        else:
            for sentence in self.sentences:
                for i in range(self.p):
                    sentence.insert(0, {"form": "<s>", "upos": "UNK"})
                for i in range(self.s):
                    sentence.append({"form": "</s>", "upos": "UNK"})
            for sentence in self.sentences:
                for i in range(self.p, len(sentence) - self.s):
                    X.append([sentence[j]["form"] for j in range(i - self.p, i + self.s + 1)])
                    y.append(sentence[i]["upos"])
            X = [[self.vocab[token] for token in x] for x in X]
            y = [self.upos_vocab[upos] if upos in self.upos_vocab else self.upos_vocab["UNK"] for upos in y]
            y = np.eye(len(self.upos_vocab))[y]
            return X, y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    
class FFNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,p, s, vocab, n_layers, activation="relu"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear((p + s + 1) * embedding_dim, hidden_dim)
        if (n_layers > 1):
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.n_layers = n_layers
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        self.vocab = vocab
        self.p = p
        self.s = s
        
    def forward(self, x):
        x = x.view(-1)
        x = self.embedding(x)
        x = x.view(-1, (self.p + self.s + 1) * self.embedding_dim)
        x = self.fc1(x)
        x = self.act(x)
        if (self.n_layers > 1):
            x = self.fc2(x)
            x = self.act(x)
        x = self.fc3(x)
        return x
    
class LSTMTagger(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers,activation="relu", bidirectionality = False):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.bidirectionality = bidirectionality
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectionality)
        if bidirectionality:
            self.hidden2tag = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, output_dim)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.bidirectionality:
            h0 = torch.zeros(self.n_layers*2, embeds.size(1), self.hidden_dim).to(device)
            c0 = torch.zeros(self.n_layers*2, embeds.size(1), self.hidden_dim).to(device)
        else:
            h0 = torch.zeros(self.n_layers, embeds.size(1), self.hidden_dim).to(device)
            c0 = torch.zeros(self.n_layers, embeds.size(1), self.hidden_dim).to(device)
        hidden = (h0, c0)
        o, (h, c) = self.lstm(embeds, hidden)
        o = self.act(o)
        preds = self.hidden2tag(o)
        return preds
    

def train_FFNN(model, train_loader, val_loader, epochs, device,lr):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_val = []
    loss_train = []
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            avg_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        loss_train.append(avg_loss / len(train_loader))
        print(f"EPOCH {epoch + 1} LOSS: {avg_loss / len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
            loss_val.append(val_loss / len(val_loader))
    return loss_train, loss_val, model

def train_LSTM(model, train_loader, val_loader, lr, device, epochs):
    model.train()
    loss_val = []
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
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                model.zero_grad()
                outputs = model(inputs)
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1, labels.shape[-1])
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            loss_val.append(val_loss/len(val_loader))
            print(f"Validation Loss: {val_loss/len(val_loader)}")
    return loss_train, loss_val, model

def test_FFNN(model, test_loader, device, upos_vocab):
    model.eval()
    crct = 0
    total = 0
    avg_loss = 0
    Tp = np.zeros(len(upos_vocab))
    Fp = np.zeros(len(upos_vocab))
    Fn = np.zeros(len(upos_vocab))
    confusion_matrix = np.zeros((len(upos_vocab), len(upos_vocab)))
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            crct += (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).sum().item()
            total += labels.size(0)
            avg_loss += loss.item()
            labels = torch.argmax(labels, 1)
            outputs = torch.argmax(outputs, 1)
            for i in range(len(upos_vocab)):
                for j in range(len(labels)):
                    if labels[j] == i:
                        if outputs[j] == i:
                            Tp[i] += 1
                        else:
                            Fn[i] += 1
                    else:
                        if outputs[j] == i:
                            Fp[i] += 1
            for i in range(len(labels)):
                confusion_matrix[labels[i]][outputs[i]] += 1

        precision = np.zeros(len(upos_vocab))
        for i in range(len(upos_vocab)):
            if Tp[i] + Fp[i] == 0:
                precision[i] = 0
            else:
                precision[i] = Tp[i] / (Tp[i] + Fp[i])
        recall = np.zeros(len(upos_vocab))
        for i in range(len(upos_vocab)):
            if Tp[i] + Fn[i] == 0:
                recall[i] = 0
            else:
                recall[i] = Tp[i] / (Tp[i] + Fn[i])
        f1 = np.zeros(len(upos_vocab))
        for i in range(len(upos_vocab)):
            if precision[i] + recall[i] == 0:
                f1[i] = 0
            else:
                f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        precision = precision[precision != 0]
        recall = recall[recall != 0]
        f1 = f1[f1 != 0]
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        accuracy = crct / total
        sum_Tp = np.sum(Tp)
        sum_Fp = np.sum(Fp)
        sum_Fn = np.sum(Fn)
        precision_micro = sum_Tp / (sum_Tp + sum_Fp)
        recall_micro = sum_Tp / (sum_Tp + sum_Fn)
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
        return avg_loss / len(test_loader), accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro, confusion_matrix

def test_LSTM(model, test_loader, device, upos_vocab):
    model.eval()  
    crct = 0
    total = 0
    avg_loss = 0
    # Tp, Fp, Fn for each class
    Tp = np.zeros(len(upos_vocab))
    Fp = np.zeros(len(upos_vocab))
    Fn = np.zeros(len(upos_vocab))
    confusion_matrix = np.zeros((len(upos_vocab), len(upos_vocab)))

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs_ = outputs.view(-1, outputs.shape[-1])
            labels_ = labels.view(-1, labels.shape[-1])
            loss = criterion(outputs_, labels_)
            avg_loss += loss.item()
            _,labels = torch.max(labels, 2)
            _, predicted = torch.max(outputs, 2)
            total += labels.size(0)*labels.size(1)
            crct += (predicted == labels).sum().item()
            labels = labels.view(-1)
            predicted = predicted.view(-1)
            for i in range(len(upos_vocab)):
                for j in range(len(labels)):
                    if labels[j] == i:
                        if predicted[j] == i:
                            Tp[i] += 1
                        else:
                            Fn[i] += 1
                    else:
                        if predicted[j] == i:
                            Fp[i] += 1
            for i in range(len(labels)):
                confusion_matrix[labels[i]][predicted[i]] += 1
        
        precision = np.zeros(len(upos_vocab))
        for i in range(len(upos_vocab)):
            if Tp[i] + Fp[i] == 0:
                precision[i] = 0
            else:
                precision[i] = Tp[i] / (Tp[i] + Fp[i])
        recall = np.zeros(len(upos_vocab))
        for i in range(len(upos_vocab)):
            if Tp[i] + Fn[i] == 0:
                recall[i] = 0
            else:
                recall[i] = Tp[i] / (Tp[i] + Fn[i])
        f1 = np.zeros(len(upos_vocab))
        for i in range(len(upos_vocab)):
            if precision[i] + recall[i] == 0:
                f1[i] = 0
            else:
                f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        precision = precision[precision != 0]
        precision = precision[precision != 0]
        recall = recall[recall != 0]
        f1 = f1[f1 != 0]
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        accuracy = crct / total
        sum_Tp = np.sum(Tp)
        sum_Fp = np.sum(Fp)
        sum_Fn = np.sum(Fn)
        precision_micro = sum_Tp / (sum_Tp + sum_Fp)
        recall_micro = sum_Tp / (sum_Tp + sum_Fn)
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
        return avg_loss / len(test_loader), accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro, confusion_matrix