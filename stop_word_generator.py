from fuzzywuzzy import fuzz

#throw out words with the highest frequency
stop_words = [
    'brokerage',
    'corp',
    'corporate',
    'advisor',
    'commercial',
    'investment',
    'organization',
    'enterpises',
    'holding',
    'development',
    'management',
    'commercial',
    'realty',
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
]
all_words = [word for sub_list in [item.split() for item in list(corpus)] for word in sub_list]
from collections import defaultdict

some_dict = defaultdict(list)
threshold_value = 75
for word in stop_words:
    for curr_word in all_words:
        if fuzz.ratio(word, curr_word) > threshold_value:
            some_dict[word].append(curr_word)
