import torch
import numpy as np
import sys

def predict_tags(sentence, method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = 1
    s = 1
    if method == "FFN":
        model = torch.load("model.pth").to(device)
        vocab = np.load("vocab.npy", allow_pickle=True).item()
        upos_vocab = np.load("upos_vocab.npy", allow_pickle=True).item()
        # preprocessing the sentence
        orig = sentence.split()
        sent = sentence.lower().split()
        sent = ["<s>"] * p + sent + ["</s>"] * s
        inputs = []
        for i in range(p, len(sent) - s):
            inputs.append([sent[j] for j in range(i - p, i + s + 1)])
        inputs = [[vocab[token] if token in vocab else vocab["<unk>"] for token in x] for x in inputs]
        inputs = torch.tensor(inputs).to(device)
        model.eval()
        outputs = model(inputs).to("cpu")
        preds = torch.argmax(outputs, 1)
        preds = preds.numpy()
        for i in range(len(orig)):
            print("{} {}".format(orig[i], list(upos_vocab.keys())[list(upos_vocab.values()).index(preds[i])]))
        
    elif method == "RNN":
        model = torch.load("lstm_model.pth").to(device)
        vocab = np.load("vocab_lstm.npy", allow_pickle=True).item()
        upos_vocab = np.load("upos_vocab_lstm.npy", allow_pickle=True).item()
        orig = sentence.split()
        sent = sentence.lower().split()
        sent = ["<s>"] + sent + ["</s>"]
        inputs = [vocab[token] if token in vocab else vocab["<unk>"] for token in sent]
        inputs = torch.tensor(inputs).to(device)
        inputs = inputs.view(1, -1)
        model.eval()
        outputs = model(inputs).to("cpu")
        preds = torch.argmax(outputs, 2)
        preds = preds.numpy()
        preds = preds[0]
        for i in range(len(orig)):
            print("{} {}".format(orig[i], list(upos_vocab.keys())[list(upos_vocab.values()).index(preds[i+1])]))

if __name__ == "__main__":

    if sys.argv[1] == "-f":
        method = "FFN"
    elif sys.argv[1] == "-r":
        method = "RNN"
    else:
        print("Invalid method")
        sys.exit()
    sentence = input("Enter a sentence: ")
    predict_tags(sentence, method)