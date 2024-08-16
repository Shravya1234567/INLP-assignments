from lm import N_Gram_Model
import sys

def generate(method, sentence, corpus,k,n):
    ngram = N_Gram_Model()
    if corpus == "pride.txt":
        ngram.load_model('pride.pkl')
        probs = ngram.generate(sentence, method, n, k)
        return probs
    else:
        ngram.load_model('ulysses.pkl')
        probs = ngram.generate(sentence, method, n, k)
        return probs
    
if __name__ == "__main__":
    method = sys.argv[1]
    corpus = sys.argv[2]
    k = sys.argv[3]
    k = int(k)
    sentence = input("input sentence: ")
    n = input("input n: ")
    n = int(n)  
    print(generate(method, sentence, corpus, k,n))
        
