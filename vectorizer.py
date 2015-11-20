from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def vectorize(corpus, model_type='bag of words', ngrams=1, tokenizer='char'):
    params = {
        'analyzer':'char',
        'ngram_range' : (1, ngrams)
    }
 
    if model_type == 'bag of words':
        vectorizer = CountVectorizer(**params)
    elif model_type ==  'tfidf':
        vectorizer = TfidfVectorizer(**params)
    feature_M = vectorizer.fit_transform(corpus)
    
    return feature_M.todense()
    