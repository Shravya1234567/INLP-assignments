from lm import N_Gram_Model

ngram = N_Gram_Model()
ngram.load_model('pride.pkl')
ngram.evaluation_test('gt', '2021101051_LM1_test-perplexity.txt')
ngram.evaluation_test('i', '2021101051_LM2_test-perplexity.txt')
ngram.evaluation_train('gt', '2021101051_LM1_train-perplexity.txt')
ngram.evaluation_train('i', '2021101051_LM2_train-perplexity.txt')

n_gram1 = N_Gram_Model()
n_gram1.load_model('ulysses.pkl')
n_gram1.evaluation_test('gt', '2021101051_LM3_test-perplexity.txt')
n_gram1.evaluation_test('i', '2021101051_LM4_test-perplexity.txt')
n_gram1.evaluation_train('gt', '2021101051_LM3_train-perplexity.txt')
n_gram1.evaluation_train('i', '2021101051_LM4_train-perplexity.txt')