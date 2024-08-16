import torch
from io import open
import numpy as np
from conllu import parse_incr
from torch.utils.data import DataLoader
from utils import CreateDataset, test_LSTM
import seaborn as sns
import matplotlib.pyplot as plt

# loading test data
data_file = open("data/en_atis-ud-test.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(data_file))
data_file.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading vocab
vocab = np.load("vocab_lstm.npy", allow_pickle=True).item()
upos_vocab = np.load("upos_vocab_lstm.npy", allow_pickle=True).item()

# creating dataset
batch_size = 1

test_data = CreateDataset(sentences,upos_vocab=upos_vocab,vocab=vocab,method='LSTM')
test_loader = DataLoader(test_data, batch_size=batch_size)

# loading validation data
data_file = open("data/en_atis-ud-dev.conllu", "r", encoding="utf-8")
sentences_val = list(parse_incr(data_file))
data_file.close()

val_data = CreateDataset(sentences_val, vocab=vocab, upos_vocab=upos_vocab, method='LSTM')
val_loader = DataLoader(val_data, batch_size=batch_size)

# loading the model
model_2 = torch.load("lstm_model.pth").to(device)

# Testing the model
model_2.eval()

loss, accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro, confusion_matrix = test_LSTM(model_2, test_loader, device, upos_vocab)
loss_val, accuracy_val, precision_macro_val, recall_macro_val, f1_macro_val, precision_micro_val, recall_micro_val, f1_micro_val, confusion_matrix_val = test_LSTM(model_2, val_loader, device, upos_vocab)

# validation results
print("Validation Results:")
print("Validation Loss: ", loss_val)
print("Validation Accuracy: ", accuracy_val)
print("Validation Precision Macro: ", precision_macro_val)
print("Validation Recall Macro: ", recall_macro_val)
print("Validation F1 Macro: ", f1_macro_val)
print("Validation Precision Micro: ", precision_micro_val)
print("Validation Recall Micro: ", recall_micro_val)
print("Validation F1 Micro: ", f1_micro_val)
print("Validation Confusion Matrix: ")
print(confusion_matrix_val)

plt.figure(figsize=(20, 10))
sns.heatmap(confusion_matrix_val, annot=True, xticklabels=upos_vocab.keys(), yticklabels=upos_vocab.keys(), cmap='Blues', fmt='f')
plt.show()

# test results
print("Test Results:")
print("Test Loss: ", loss)
print("Test Accuracy: ", accuracy)
print("Test Precision Macro: ", precision_macro)
print("Test Recall Macro: ", recall_macro)
print("Test F1 Macro: ", f1_macro)
print("Test Precision Micro: ", precision_micro)
print("Test Recall Micro: ", recall_micro)
print("Test F1 Micro: ", f1_micro)
print("Test Confusion Matrix: ")
print(confusion_matrix)

plt.figure(figsize=(20, 10))
sns.heatmap(confusion_matrix, annot=True, xticklabels=upos_vocab.keys(), yticklabels=upos_vocab.keys(), cmap='Blues', fmt='f')
plt.show()