import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from string import punctuation
from fuzzywuzzy import fuzz
import cPickle


def compute_similarity(s1, s2):
    return 1.0 - (0.01 * max(
        fuzz.ratio(s1, s2),
        fuzz.token_sort_ratio(s1, s2),
        fuzz.token_set_ratio(s1, s2)))


def compute_matrix(master_table):
    stitles = list(master_table['name_clean'])
    X = np.zeros((len(stitles), len(stitles)))
    for i in range(len(stitles)):
        if i > 0 and i % 10 == 0:
            print "Processed %d/%d rows of data" % (i, X.shape[0])
        for j in range(len(stitles)):
            if X[i, j] == 0.0:        
                X[i, j] = compute_similarity(stitles[i].lower(), stitles[j].lower())
                X[j, i] = X[i, j]
    return X

stop_words = [
    'commercial',
    'realty',
    'real',
    'estate',
    'group',
    'properties',
    'company',
    'associates',
    'partners',
    'inc',
    'co',
    'services',
    'llc',
    'property',
    'the'
]


def remove_punc(t_string):
    check = lambda char: ' ' if char in set(punctuation) else char
    return ''.join(map(check, list(t_string)))
        
    
def preprocess_str(target_string):
    
    #lower case
    target_string = target_string.lower()
    
    #remove punctuation first
    nopunc_target_string = remove_punc(target_string)
    
    #remove stop words
    no_target_string = ' '.join([word for word in nopunc_target_string.split() if word not in stop_words])
    
    return no_target_string

def encode(str_):
    output=[]
    for char in str_:
        try:
            output.append(char.encode('utf-8'))
        except:
            output.append(' ')
    return ''.join(output)


def clean(name):
#     if len(name)<=2: return None
    
    #remove spaces
    name = name.strip()
    
    #strip out
    name = name.replace('\"', '')
    
    name = encode(name)
    
    return name
if __name__ == '__main__':
    stops = set(stopwords.words('english'))
    with open('realty_company_no_newline.csv','rb') as f:
        super_list = []
        for i, line in enumerate(f):
            id_, ver, names_str = line.strip().split(',',2)
            super_list.append([id_, ver, clean(names_str)])

    df_company = pd.DataFrame(super_list[1:], columns=super_list[0])
    df_company['name_clean'] = df_company['name'].apply(preprocess_str)
    X = compute_matrix(df_company)
    
    with open('X.pkl','wb') as f:
        cPickle.dump(X, f)
