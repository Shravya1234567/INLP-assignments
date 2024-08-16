import torch
from io import open
import numpy as np
from conllu import parse_incr
from torch.utils.data import DataLoader
from utils import CreateDataset, FFNN, train_FFNN
import wandb
import pandas as pd

# variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cut_off_freq = 3
p = 2
s = 2
batch_size = 32

# loading saved vocab and upos_vocab
vocab = np.load("vocab.npy", allow_pickle=True).item()
upos_vocab = np.load("upos_vocab.npy", allow_pickle=True).item()

# loading the train data
data_file = open("data/en_atis-ud-train.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(data_file))
data_file.close()

# train data
train_data = CreateDataset(sentences, p, s, cut_off_freq, upos_vocab, vocab)

# loading the validation data
data_file = open("data/en_atis-ud-dev.conllu", "r", encoding="utf-8")
sentences = list(parse_incr(data_file))
data_file.close()

# validation data
val_data = CreateDataset(sentences, p, s, cut_off_freq, upos_vocab, vocab)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

sweep_config = {
    "name": "model_1_tuning",
    "method": "grid",
    "parameters": {
        "n_layers": {"values": [1, 2, 3]},
        "embedding_dim": {"values": [64, 128, 256]},
        "hidden_dim": {"values": [64, 128, 256]},
        "lr": {"values": [0.001, 0.01, 0.1]},
        "epochs": {"values": [5, 10, 15, 20]},
        "activation": {"values": ["relu", "tanh"]},
    },
    "metric": {"goal": "minimize", "name": "val_loss"}
}

sweep_id = wandb.sweep(sweep_config, project="INLP-Assignment-2")

def train():

    df = pd.DataFrame(columns=["val_loss", "loss_train", "n_layers", "embedding_dim", "hidden_dim", "lr", "epochs", "activation"])
    
    config_defaults = {
        "n_layers": 1,
        "embedding_dim": 128,
        "hidden_dim": 128,
        "lr": 0.001,
        "epochs": 10,
        "activation": "relu"
    }
    with wandb.init(config=config_defaults):
        config = wandb.config
        model = FFNN(len(vocab), config.embedding_dim, config.hidden_dim, len(upos_vocab), p, s, vocab, config.n_layers,config.activation).to(device)
        loss_train, loss_val, model = train_FFNN(model, train_loader, val_loader, config.epochs, device, config.lr)
        wandb.log({"val_loss": loss_val[-1], "loss_train": loss_train[-1], "n_layers": config.n_layers, "embedding_dim": config.embedding_dim, "hidden_dim": config.hidden_dim, "lr": config.lr, "epochs": config.epochs, "activation": config.activation})
        df = df._append({"val_loss": loss_val[-1], "loss_train": loss_train[-1], "n_layers": config.n_layers, "embedding_dim": config.embedding_dim, "hidden_dim": config.hidden_dim, "lr": config.lr, "epochs": config.epochs,"activation": config.activation}, ignore_index=True)
    df = df.sort_values(by="val_loss")
    df.to_csv("model_1_tuning.csv", index=False)
   
wandb.agent(sweep_id, train)
