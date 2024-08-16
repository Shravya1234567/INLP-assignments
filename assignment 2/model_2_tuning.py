import torch
from io import open
import numpy as np
from conllu import parse_incr
from torch.utils.data import DataLoader
from utils import CreateDataset, LSTMTagger, train_LSTM
import wandb

# variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cut_off_freq = 3
batch_size = 32

# loading saved vocab and upos_vocab
vocab = np.load("vocab_lstm.npy", allow_pickle=True).item()
upos_vocab = np.load("upos_vocab_lstm.npy", allow_pickle=True).item()

# loading the train data
data_file = open("data/en_atis-ud-train.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(data_file))
data_file.close()

# train data
train_data = CreateDataset(sentences,upos_vocab=upos_vocab,vocab=vocab,method='LSTM')

# loading the validation data
data_file = open("data/en_atis-ud-dev.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(data_file))
data_file.close()

# validation data
val_data = CreateDataset(sentences,upos_vocab=upos_vocab,vocab=vocab,method='LSTM')

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

sweep_config = {
    "name": "model_2_tuning",
    "method": "grid",
    "parameters": {
        "n_layers": {"values": [1, 2]},
        "embedding_dim": {"values": [64, 128, 256]},
        "hidden_dim": {"values": [64, 128, 256]},
        "lr": {"values": [0.001, 0.01]},
        "epochs": {"values": [5, 10, 15]},
        "activation": {"values": ["relu", "tanh"]},
        "bidirectionality": {"values": [True, False]}
    },
    "metric": {"goal": "minimize", "name": "val_loss"}
}

sweep_id = wandb.sweep(sweep_config, project="NLP-Assignment-2")

def train():
    
    config_defaults = {
        "n_layers": 1,
        "embedding_dim": 128,
        "hidden_dim": 128,
        "lr": 0.001,
        "epochs": 10,
        "activation": "relu",
        "bidirectionality": False
    }
    with wandb.init(config=config_defaults):
        config = wandb.config
        model = LSTMTagger(len(vocab), config.embedding_dim, config.hidden_dim, len(upos_vocab), config.n_layers, config.activation, config.bidirectionality).to(device)
        loss_train, loss_val, model = train_LSTM(model, train_loader, val_loader, config.lr, device, config.epochs)
        wandb.log({"val_loss": loss_val[-1], "loss_train": loss_train[-1], "n_layers": config.n_layers, "embedding_dim": config.embedding_dim, "hidden_dim": config.hidden_dim, "lr": config.lr, "epochs": config.epochs, "activation": config.activation, "bidirectionality": config.bidirectionality})
   
wandb.agent(sweep_id, train)
