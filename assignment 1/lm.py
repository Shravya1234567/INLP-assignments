from tokenizer import tokenize
import contractions
import random
from scipy.stats import linregress
import numpy as np
import pickle


def preprocess(text):
    text = text.lower()
    text_ = contractions.fix(text)
    return text_

class N_Gram_Model:
    def __init__(self, n=None, corpus_path=None):
        self.n = n
        self.corpus_path = corpus_path
        self. train_corpus = [[]]
        self.test_corpus = [[]]
        self.trigrams = {}
        self.bigrams = {}
        self.unigrams = {}
        self.tot_unigrams = 0
        self.tot_bigrams = 0
        self.tot_trigrams = 0
        self.probabilities = {}
        self.probabilities_gt = {}
        self.probabilities_i_tri = {}
        self.probabilities_i_bi = {}
        self.probabilities_i_uni = {}
        self.a = None
        self.b = None
        self.Nr = {}
        self.r_star = {}
        self.lambdas = []

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        self.n = model.n
        self.corpus_path = model.corpus_path
        self.train_corpus = model.train_corpus
        self.test_corpus = model.test_corpus
        self.trigrams = model.trigrams
        self.bigrams = model.bigrams
        self.unigrams = model.unigrams
        self.tot_unigrams = model.tot_unigrams
        self.tot_bigrams = model.tot_bigrams
        self.tot_trigrams = model.tot_trigrams
        self.probabilities = model.probabilities
        self.probabilities_gt = model.probabilities_gt
        self.probabilities_i_tri = model.probabilities_i_tri
        self.probabilities_i_bi = model.probabilities_i_bi
        self.probabilities_i_uni = model.probabilities_i_uni
        self.a = model.a
        self.b = model.b
        self.Nr = model.Nr
        self.r_star = model.r_star
        self.lambdas = model.lambdas
        
    def get_tokens(self, corpus):
        corpus = preprocess(corpus)
        tokens = tokenize(corpus)
        for i in range(len(tokens)):
            tokens[i] = [token for token in tokens[i] if any(c.isalpha() for c in token)]
        if self.n != 1:
            for i in range(len(tokens)):
                tokens[i] = ['<s>'] * (self.n-1) + tokens[i] + ['</s>']
        return tokens
    
    def read_file(self): 
        with open(self.corpus_path, 'r') as f:
            corpus = f.read().replace('\n', ' ')
        tokens = self.get_tokens(corpus)
        return tokens
    
    def setup(self):
        tokens = self.read_file()
        test_tokens = []
        for i in range(1000):
            test_tokens.append(tokens.pop(random.randrange(len(tokens))))
        train_tokens = tokens
        self.test_corpus = test_tokens
        self.train_corpus = train_tokens

    def add_unks(self):
        unigrams = {}
        for sentence in self.train_corpus:
            for word in sentence:
                if word not in unigrams:
                    unigrams[word] = 1
                else:
                    unigrams[word] += 1

        for sentence in self.train_corpus:
            for i in range(len(sentence)):
                if unigrams[sentence[i]]  <= 1:
                    sentence[i] = '<unk>'
    
    def train(self):
        self.setup()
        self.add_unks()
        self.trigrams = {}
        self.bigrams = {}
        self.unigrams = {}
        self.tot_unigrams = 0
        self.tot_bigrams = 0
        self.tot_trigrams = 0

        for sentence in self.train_corpus:
            for i in range(len(sentence)-2):
                trigram = tuple(sentence[i:i+3])
                if trigram in self.trigrams:
                    self.trigrams[trigram] += 1
                else:
                    self.trigrams[trigram] = 1
                self.tot_trigrams += 1
            for i in range(len(sentence)-1):
                bigram = tuple(sentence[i:i+2])
                if bigram in self.bigrams:
                    self.bigrams[bigram] += 1
                else:
                    self.bigrams[bigram] = 1
                self.tot_bigrams += 1
            for word in sentence:
                if word in self.unigrams:
                    self.unigrams[word] += 1
                else:
                    self.unigrams[word] = 1
                self.tot_unigrams += 1
        probabilities = {}
        for trigram in self.trigrams:
            bigram = tuple(trigram[:2])
            probabilities[trigram] = self.trigrams[trigram]/self.bigrams[bigram]
        self.probabilities = probabilities
        self.probabilities_gt = self.good_turing_smoothing()
        self.probabilities_i_tri, self.probabilities_i_bi, self.probabilities_i_uni = self.interpolated_probabilities()
        return probabilities, self.probabilities_gt,self.probabilities_i_tri, self.probabilities_i_bi, self.probabilities_i_uni
    
    def calculate_Nr(self):
        Nr = {}
        for trigram in self.trigrams:
            if self.trigrams[trigram] in Nr:
                Nr[self.trigrams[trigram]] += 1
            else:
                Nr[self.trigrams[trigram]] = 1
        return Nr
    
    def calculate_zr(self):
        zr = {}
        sorted_counts = sorted(self.Nr.keys())
        for r in sorted_counts:
            if r == 1:
                t = sorted_counts[1]
                zr[r] = self.Nr[r] /(1/2 * t)
            elif r == sorted_counts[-1]:
                q = sorted_counts[-2]
                zr[r] = self.Nr[r] / (r - q)
            else:
                q = sorted_counts[sorted_counts.index(r)-1]
                t = sorted_counts[sorted_counts.index(r)+1]
                zr[r] = self.Nr[r] / (1/2 * (t-q))
        return zr
    
    def linear_regression(self,zr):
        x = []
        y = []
        for r in zr:
            x.append(r)
            y.append(zr[r])
        x = np.log(x)
        y = np.log(y)
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        self.a = intercept
        self.b = slope

    def smoothed_Nr(self):
        Nr = {}
        r_last = max(self.Nr.keys())
        for r in range(r_last+1):
            if r not in self.Nr:
                Nr[r] = 0
        for r in range(r_last+1):
            if r == 0:
                Nr[r] = 1
            elif r < 5:
                Nr[r] = self.Nr[r]
            else:
                Nr[r] = np.exp(self.a + self.b * np.log(r))
        Nr[r_last+1] = np.exp(self.a + self.b * np.log(r_last+1) )
        return Nr, r_last
    
    def good_turing_smoothing(self):
        self.Nr = self.calculate_Nr()
        zr = self.calculate_zr()
        self.linear_regression(zr)
        Nr,r_last = self.smoothed_Nr()
        r_star = {}
        for r in range(r_last+1):
            r_star[r] = (r+1) * Nr[r+1] / Nr[r]
        probabilities_gt = {}
        for trigram in self.trigrams:
            numerator = r_star[self.trigrams[trigram]]
            denominator = 0
            for unigram in self.unigrams:
                x = trigram[:2]
                x = tuple(x + (unigram,))
                if x in self.trigrams:
                    denominator += r_star[self.trigrams[x]]
                else:
                    denominator += r_star[0]
            probabilities_gt[trigram] = numerator/denominator
        self.r_star = r_star
        self.probabilities_gt = probabilities_gt
        return probabilities_gt
    
    def get_lambdas(self):
        lambdas = []
        for i in range(self.n):
            lambdas.append(0)
        for trigram in self.trigrams:
            # t1, t2, t3
            bigram = tuple(trigram[:2]) # t1, t2
            bigram_ = tuple(trigram[-2:]) # t2, t3
            unigram = trigram[-2] # t2
            unigram_ = trigram[-1] # t3
            if self.bigrams[bigram] != 1:
                a = (self.trigrams[trigram]-1)/(self.bigrams[bigram]-1)
            else:
                a = 0
            if self.unigrams[unigram] != 1:
                b = (self.bigrams[bigram_]-1)/(self.unigrams[unigram]-1)
            else:
                b = 0
            c = (self.unigrams[unigram_]-1)/(self.tot_unigrams-1)
            if a >= b and a >= c:
                lambdas[0] += self.trigrams[trigram]
            elif b >= a and b >= c:
                lambdas[1] += self.trigrams[trigram]
            else:
                lambdas[2] += self.trigrams[trigram]
        
        lambdas[0] /= self.tot_trigrams
        lambdas[1] /= self.tot_trigrams
        lambdas[2] /= self.tot_trigrams
        return lambdas
    
    def interpolated_probabilities(self):
        self.lambdas = self.get_lambdas()
        probabilities_i_tri = {}
        probabilities_i_bi = {}
        probabilities_i_uni = {}
        for trigram in self.trigrams:
            # t1, t2, t3
            bigram = tuple(trigram[:2]) # t1, t2
            bigram_ = tuple(trigram[-2:]) # t2, t3
            unigram = trigram[-2] # t2
            unigram_ = trigram[-1] # t3
            probabilities_i_tri[trigram] = self.lambdas[0] * self.trigrams[trigram]/self.bigrams[bigram] + self.lambdas[1] * self.bigrams[bigram_]/self.unigrams[unigram] + self.lambdas[2] * self.unigrams[unigram_]/self.tot_unigrams
        for bigram in self.bigrams:
            # t1, t2
            unigram = bigram[0] # t1
            unigram_ = bigram[1] # t2
            probabilities_i_bi[bigram] = self.lambdas[1] * self.bigrams[bigram]/self.unigrams[unigram] + self.lambdas[2] * self.unigrams[unigram_]/self.tot_unigrams
        for unigram in self.unigrams:
            probabilities_i_uni[unigram] = self.lambdas[2] * self.unigrams[unigram]/self.tot_unigrams
        self.probabilities_i_tri = probabilities_i_tri
        self.probabilities_i_bi = probabilities_i_bi
        self.probabilities_i_uni = probabilities_i_uni
        return probabilities_i_tri, probabilities_i_bi, probabilities_i_uni
    
    def perplexity(self, sentence, method):
        if method == 'gt':
            log_sum = 0
            for i in range(len(sentence)-2):
                trigram = tuple(sentence[i:i+3])
                if trigram in self.probabilities_gt:
                    log_sum += np.log(self.probabilities_gt[trigram])
                else:
                    numerator = self.r_star[0]
                    denominator = 0
                    for unigram in self.unigrams:
                        x = trigram[:2]
                        x = tuple(x + (unigram,))
                        if x in self.trigrams:
                            denominator += self.r_star[self.trigrams[x]]
                        else:
                            denominator += self.r_star[0]
                    log_sum += np.log(numerator/denominator)                   
            perplexity = np.exp(-log_sum/(len(sentence)-2))
        elif method == 'i':
            log_sum = 0
            for i in range(len(sentence)-2):
                trigram = tuple(sentence[i:i+3])
                bigram = tuple(trigram[1:])
                unigram = trigram[2]
                if trigram in self.probabilities_i_tri:
                    log_sum += np.log(self.probabilities_i_tri[trigram])
                elif bigram in self.probabilities_i_bi:
                    log_sum += np.log(self.probabilities_i_bi[bigram])
                else:
                    log_sum += np.log(self.probabilities_i_uni[unigram])
            perplexity = np.exp(-log_sum/(len(sentence)-2))
        else:
            log_sum = 0
            for i in range(len(sentence)-2):
                trigram = tuple(sentence[i:i+3])
                if trigram in self.probabilities:
                    log_sum += np.log(self.probabilities[trigram])
                else:
                    log_sum += np.log(10**-15)
            perplexity = np.exp(-log_sum/(len(sentence)-2))

        return perplexity
    
    def evaluation(self,method, sentence):
        sentence = preprocess(sentence)
        sentence = tokenize(sentence)
        for i in range(len(sentence)):
            sentence[i] = [token for token in sentence[i] if any(c.isalpha() for c in token)]
        if self.n != 1:
            for i in range(len(sentence)):
                sentence[i] = ['<s>'] * (self.n-1) + sentence[i] + ['</s>']
        for i in range(len(sentence)):
            for j in range(len(sentence[i])):
                if sentence[i][j] not in self.unigrams:
                    sentence[i][j] = '<unk>'
        perplexity = self.perplexity(sentence[0],method)
        return perplexity
                
                
    def evaluation_test(self,method, file_path):
        perplexities = []
        for sentence in self.test_corpus:
            for i in range(len(sentence)):
                if sentence[i] not in self.unigrams:
                    sentence[i] = '<unk>'
        for sentence in self.test_corpus:
            perplexities.append(self.perplexity(sentence,method))
        print(np.mean(perplexities))
        with open(file_path, 'w') as f:
            f.write('avg perplexity: ' + str(np.mean(perplexities)) + '\n')
            for i in range(len(self.test_corpus)):
                f.write(' '.join(self.test_corpus[i]) + '\t' + str(perplexities[i]) + '\n')
        return perplexities
    
    def evaluation_train(self,method, file_path):
        perplexities = []
        for sentence in self.train_corpus:
            perplexities.append(self.perplexity(sentence,method))
        with open(file_path, 'w') as f:
            f.write('avg perplexity: ' + str(np.mean(perplexities)) + '\n')
            for i in range(len(self.train_corpus)):
                f.write(' '.join(self.train_corpus[i]) + '\t' + str(perplexities[i]) + '\n')
        return perplexities
    
    def generate_n_grams(self,n):
        if n == 1:
            return self.unigrams, self.tot_unigrams
        elif n == 2:
            return self.bigrams, self.tot_bigrams
        elif n == 3:
            return self.trigrams, self.tot_trigrams
        else:
            tot = 0
            n_grams = {}
            for sentence in self.train_corpus:
                for i in range(len(sentence)-n+1):
                    n_gram = tuple(sentence[i:i+n])
                    if n_gram in n_grams:
                        n_grams[n_gram] += 1
                    else:
                        n_grams[n_gram] = 1
                    tot += 1
            return n_grams, tot
    
    def generate(self, sentence, method, n, k):
        n_grams,tot = self.generate_n_grams(n)
        sentence = preprocess(sentence)
        sentence = tokenize(sentence)
        for i in range(len(sentence)):
            sentence[i] = [token for token in sentence[i] if any(c.isalpha() for c in token)]
        if self.n != 1:
            for i in range(len(sentence)):
                if i == len(sentence)-1:
                    sentence[i] = ['<s>'] * (self.n-1) + sentence[i]
                else:
                    sentence[i] = ['<s>'] * (self.n-1) + sentence[i]+ ['</s>']
        for i in range(len(sentence)):
            for j in range(len(sentence[i])):
                if sentence[i][j] not in self.unigrams:
                    sentence[i][j] = '<unk>'
        if method == 'i':
            sentence = sentence[-1]
            sentence = sentence[-2:]
            probs = {}
            for unigram in self.unigrams:
                x = list(sentence)  
                x.append(unigram)
                x = tuple(x) 
                y = list(sentence[-1:])
                y.append(unigram)
                y = tuple(y)
                if x in self.trigrams:
                    probs[unigram] = self.probabilities_i_tri[x]
                elif y in self.bigrams:
                    probs[unigram] = self.probabilities_i_bi[y]
                else:
                    probs[unigram] = self.probabilities_i_uni[unigram]
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            return sorted_probs[:k]
        else:
            sentence = sentence[-1]
            sentence = sentence[-n+1:]
            if n==1:
                probs = {}
                for unigram in self.unigrams:
                    if unigram in n_grams:
                        probs[unigram] = n_grams[unigram]/tot
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                return sorted_probs[:k]
            probs = {}
            for unigram in self.unigrams:
                x = list(sentence)
                x.append(unigram)
                x = tuple(x)
                if x in n_grams:
                    probs[unigram] = n_grams[x]/tot
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            return sorted_probs[:k]