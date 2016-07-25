from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LSHForest
from annoy import Annoy

from fuzzywuzzy import fuzz

import pandas as pd
from scipy.stats import skew
from scipy.stats import boxcox
from sklearn import preprocessing
import matplotlib.pyplot as plt




import time
from functools import wraps

PROF_DATA = {}

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling

def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print "Function %s called %d times. " % (fname, data[0]),
        print 'Execution time max: %.3f, average: %.3f' % (max_time, avg_time)

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}

# need to add a class for graph theory
# nn class
class Deduper_NN(object):
    '''
    DESIGN of this class
    I need to re-evaluate whether or not I want the state of the model/vector space 
    being saved in the event that I dont' I should just kill the self.model = model.fit() stuff
    and pass parameters from one function to another.

    *Figure out a rigorous statistical way to measure quality of NN tree. 
    Would it be whatever's skewed to the left?... how can you gaurantee that it's clustered well?
    
    methods
    ------
    train
        - model type
    predict
    preprocess
        - various stuff
    '''

    metrics = [
        'cosine', 
        'euclidean',
        'dice', 
        'jaccard', 
        'braycurtis',
        'canberra', 
    ]
    
    vector_space = None

    def read_in_the_file(self, file_name):
        
        #read in subject file
        with open(file_name) as f:
             self.orig_file = [line.strip() for line in f]
    
    def build_vectorizer(self, corpus, model_type='bag of words', ngrams=1, tokenizer='char'):
        '''
        *add word2vec
        '''
        
        #think of params
        params = {
            'analyzer': tokenizer,
            'ngram_range' : (1, ngrams)
        }
    
        if model_type == 'bag of words':
            vectorizer = CountVectorizer(**params)
        elif model_type ==  'tfidf':
            vectorizer = TfidfVectorizer(**params)
        
        self.vector_space = vectorizer.fit_transform(corpus) 
        self.vectorizer = vectorizer 
    
    def find_all_duplicates(self):
        
        #find all duplicates
        all_dups_dict = {idx : self.predict(line) for idx, line in enumerate(self.orig_file)}
        return all_dups_dict
    
    def fit_model(self, model_type='brute', params=None):
        '''
        fits model operating under the assumption that there's a model already built
        '''

        if model_type == 'brute':
            self.model = NearestNeighbors(algorithm='brute', **params)
        elif model_type == 'lsh':
            self.model = LSHForest( **params)
        # elif model_type == 'annoy':
        #     self.model = Annoy(**params)

        self.model.fit(self.vector_space)
        print self.model        

    def predict(self, new_data_pt, radius_threshold=.25):
        '''
        not sure how to find the optimal threshold here
        '''
        #careful to note that it takes a single string and converts to a list object of strings
        pt = self.vectorizer.transform([new_data_pt])
        
        #how to find optimal radius?
        distance_from_origin, indices = self.model.radius_neighbors(pt, radius=radius_threshold)
        
        #unpacking
        distance_from_origin = distance_from_origin[0]
        indices = indices[0]

        grabbing_the_lines_from_file = [self.orig_file[index] for index in indices]

        return grabbing_the_lines_from_file
    
    def grid_search(self):
        '''
        I: target string
        O: prints all combinations of comparisons
        
        * this goes in the master deduper class
        '''
        
        #preprocessing variables
            #spaces or no spaces
            #combinations there of.
        
        vector_space_params = {
            #fit the vector-space
            #char-grams, words
                #unagrams, bigrams, tri-grams

            #or some combination there of, to do this we need to output and concat
            
            'model_type' : ['bag of words', 'tfidf'], #add lsi and word2vec
            'ngrams' : [1,2,3,4],
            'tokenizer' : ['char', 'word'],
        }
        
        
        
        #model selection
        model_params = {
            #add annoy later
            #build out a wrapper for the class to make it more like sciki

            #add lsh later
            #need to build a seperate parameters dict for it.

        
            'model_type' : [ 'brute']
            #fill the rest in later
        }
        
         
        #distances
        metrics = [
            # work for sparse input
            'cosine', 
            'euclidean',
            'l1',
            'l2',
            'manhattan',

            # do not work for sparese input
            # 'dice', 
            # 'jaccard', 
            # 'braycurtis',
            # 'canberra', 
            # 'mahalanobis', # this is supposed to be the shit for outlier detection
        ]
        
        
        all_params = {
            'preprocessing': None,
            'vector_space': vector_space_params,
            'nn_algo': model_params,
        }
        
        

        for nn_algo in all_params['nn_algo']['model_type']:
            for vector_space_model in all_params['vector_space']['model_type']:
                for gram in  all_params['vector_space']['ngrams']:
                    for type_of_tokenizer in  all_params['vector_space']['tokenizer']:
                        for dist_metric in metrics:
                            
                            nn_model_params = {
                                # 'model_type' : nn_algo,
                                'metric' : dist_metric,
                            }

                            vectorizer_params = {
                                'model_type' : vector_space_model,
                                'tokenizer' : type_of_tokenizer,
                                'ngrams' : gram
                            }

                            self.build_vectorizer(self.orig_file, **vectorizer_params)
                            self.fit_model(nn_algo, nn_model_params)
                            hist_arr = self.make_hist()
                            print_prof_data()
                            clear_prof_data()
                            self.plot_histogram(hist_arr)

                            
                
        
        #how do you gauge the quality of matches?
        
        pass
    
    #since this isn't a nn search model it belongs in the biggest deduper
    def brute_force_deduper(self, list_of_strings, comparison_algo, threshold=None):
        '''
        I: self explanatory
        O: dictionary {string: sorted list of matches}
        '''
        big_bag = {}
        #to deep copy or not to deep copy

        for index, s1 in enumerate(list_of_strings):
            small_bag = get_all_comparisons(list_of_strings[index:], comparison_algo)
            big_bag[s1] = sorted(small_bag, key=lambda x: x[0], reverse=True)

        return big_bag
    
    @profile
    def make_hist(self):
        '''
        these queries take while

        *add timer bit
        '''
        import sys

        #use a numpy array since the size is already pre-defined
        hist_bag = []
        
        print 'set size -- ', self.vector_space.shape[0]

        for l, observation in enumerate(self.vector_space):

            #just a way to keep track of where it's at
            if l % 30 == 0: 
                sys.stdout.write(str(l))
            

            dist, idx = self.model.kneighbors(observation, n_neighbors=2)
            dist, idx = dist[0], idx[0]

            #operating under the assumption that 
            #the first one is might be a good thing

            #find out which position the current index is in
            # remove_this_arg = [k for k, i in enumerate(idx) if i == index]
            # dist = [i for k, i in enumerate(dist) if i != remove_this_arg[0]]
            


            hist_bag.append(dist[1])


        return pd.Series(hist_bag)

    def plot_histogram(self, histogram_arr, text_pos=None):
        
        figure = plt.figure(figsize=(10,5))
        plt.hist(histogram_arr, bins=50, alpha=0.75) 
        plt.title("not scaled") 
        if text_pos:
            self.distribution_stats_text_label(text_pos[0], text_pos[1], histogram_arr)
        plt.show()


    def distribution_stats_text_label(self, position_x, position_y, data):
        label_position_decrement = 0.08 * position_y
        plt.text(position_x, position_y, "Skewness: {0:.2f}".format(skew(data))) 
        plt.text(position_x, position_y - label_position_decrement, "Mean: {0:.2f}".format(data.mean())) 
        plt.text(position_x, position_y - 2 * label_position_decrement, "Std: {0:.2f}".format(data.std())) 
        return None

    def get_all_comparisons(self, main_str, strings, comparison_algo, threshold=None):
        '''
        I: string, list of strings, string comparison algo eg. levenshtien, threshold
        O: list of tuples (match score, weight) 
        
        Takes a target string and compares it to the rest of strings in the list
        USE --

        get_all_comparisons('check', ['check1'], fuzz.ratio) 
        >>> [(91, 'check1')]
        '''
        match_bag = []

        for str_ in strings:
            match_rating = comparison_algo(main_str, str_)

            if threshold:
                if match_rating > threshold:
                    match_bag.append((match_rating, str_))
            else:
                match_bag.append((match_rating, str_))

        return match_bag




if __name__ == '__main__':
    deduper = Deduper_NN()
    deduper.read_in_the_file('target.txt')
    deduper.build_vectorizer(deduper.orig_file)
    deduper.fit_model()
    print 'model fitted'
    histo = deduper.make_hist()
    

    # for i in range(10):
    #     print '=' * 50
    #     print '\n'.join(deduper.predict(deduper.orig_file[i], 3))
