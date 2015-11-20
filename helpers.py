from string import punctuation
from stop_words_conversion_tables import company_name_conversions
from stop_words_conversion_tables import stop_words_states, initial_stop_words, additional_stop_words, nltk_stop_words


def encode(str_):
    output=[]
    for char in str_:
        try:
            output.append(char.encode('utf-8'))
        except:
            output.append(' ')
    
    return ''.join(output)


def clean(name):
    #remove spaces
    name = name.strip()
    
    #strip out
    name = name.replace('\"', '')    
    name = encode(name)    
    
    return name

def remove_punc(t_string):
    check = lambda char: ' ' if char in set(punctuation) else char
    
    return ''.join(map(check, list(t_string)))
        
def preprocess_str(target_string):
    #lower case
    target_string = target_string.lower()
    
    #convert company names
    coverted_names_target_string = ' '.join(map(lambda x: company_name_conversions.get(x) \
                                            if company_name_conversions.get(x) else x,\
                                            [word for word in target_string.split()]))
    #remove punctuation first
    nopunc_target_string = remove_punc(coverted_names_target_string)
    
    #remove stop words
    stop_words_set = set(nltk_stop_words + stop_words_states + initial_stop_words + additional_stop_words)
    no_target_string = ' '.join([word for word in nopunc_target_string.split() if word not in stop_words_set])
    
    return no_target_string
