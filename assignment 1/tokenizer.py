import re

def tokenize(text):
    
    sent_tokenizer = re.compile(r'(?<!\w\.\w.\s)(?<![A-Z]\.\s)(?<!\s\w\.\s)(?<!mr\.\s)(?<!ms\.\s)(?<!dr\.\s)(?<!mrs\.\s)(?<!Mr\.\s)(?<!Ms\.\s)(?<!Dr\.\s)(?<!Mrs\.\s)(?<=\.[\"\s]|\?[\"\s]|\![\"\s])')
    word_tokenizer = re.compile(r'\S*\w+\S*')
    number_tokenizer = re.compile(r'\d+')
    email_tokenizer = re.compile(r'\w+@\w+\.\w+')
    url_tokenizer = re.compile(r'\S*https?://\S+|\S*www\.\S+')
    hashtag_tokenizer = re.compile(r'#\w+')
    mention_tokenizer = re.compile(r'@\w+')
    punctuation_tokenizer = re.compile(r'\w\.\w.|[A-Z]\.|[a-z]\.|[Mm]r\.|[Mm]s\.|[Mm]rs\.|[Dd]r\.|<URL>|<MAILID>|<NUM>|<HASHTAG>|<MENTION>|\w+(?:-\w+)?|[^\w\s]')
    

    sentences = sent_tokenizer.split(text)
    tokens = []
    for sentence in sentences:
        if sentence and sentence != ' ':
            tokens.append(word_tokenizer.findall(sentence))
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            if number_tokenizer.match(tokens[i][j]):
                tokens[i][j] = '<NUM>'
            elif email_tokenizer.match(tokens[i][j]):
                tokens[i][j] = '<MAILID>'
            elif url_tokenizer.match(tokens[i][j]):
                tokens[i][j] = '<URL>'
            elif hashtag_tokenizer.match(tokens[i][j]):
                tokens[i][j] = '<HASHTAG>'
            elif mention_tokenizer.match(tokens[i][j]):
                tokens[i][j] = '<MENTION>'
    tokenized_text = []
    for sentence in tokens:
        t = []
        for token in sentence:
            t += punctuation_tokenizer.findall(token)
        tokenized_text.append(t)

    
    return tokenized_text

if __name__ == '__main__':
    text = input('your text: ')
    tokens = tokenize(text)
    print('tokenized text:', tokens)