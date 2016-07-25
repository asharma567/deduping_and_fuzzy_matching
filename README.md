# deduping_and_fuzzy_matching


Practical use-case of utlitizing a combination of string matching functions and DB-SCAN to deduplicate a data set.

Data-set: text data, company names

deduper_class.py: a class of Nearest Neighbor algorithms built to find the best approach for your particular use case. Also comes with an evaluation method.

grid-search: before the DB-SCAN there are multiple options of vectorizing the data-set to a BoW or a TF-IDF model tokenizing by character-grams (unigrams, bigrams). In any case, there are many hyper-parameters to play around with so I made a grid-searching script to aid in this area.



