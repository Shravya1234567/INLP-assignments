from lm import N_Gram_Model
import sys

def lang_model(corpus, method, sentence):
    ngram = N_Gram_Model()
    if corpus == "pride.txt":
        ngram.load_model('pride.pkl')
        perplexity = ngram.evaluation(method, sentence)
        return perplexity
    else:
        ngram.load_model('ulysses.pkl')
        perplexity = ngram.evaluation(method, sentence)
        return perplexity
    

if __name__ == "__main__":
    method = sys.argv[1]
    corpus = sys.argv[2]
    sentence = input("input sentence: ")
    print(lang_model(corpus, method, sentence))
