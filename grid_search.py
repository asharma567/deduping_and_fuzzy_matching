
from sklearn.cluster import DBSCAN
import pandas as pd

'''
Cluster qaulity-- 
    tfidf--
        canberra 1, 0.8
        Braycurtis .09
        jaccard .06
        euclidean 1
        manhattan 1

    tfidf bigrams--
        cosine 0.2 * i like this
        euclidean 0.6
        braycurtis 0.2


    bag trigrams--
        cosine 0.2

    bag bigrams--
        cosine 0.09
        braycurtis 0.2
        jaccard 0.4, 0.2 * i like this
        dice 0.2

    bag--
        Dice 0.06
        Jaccard 0.06
        Braycurtis 0.09
'''



def count_of_large_clusters(clusters):
    is_cluster = lambda x: 1 if x > 10 else 0
    total_cnt = sum([is_cluster(v) for k,v in clusters.value_counts().to_dict().items()])
    return total_cnt


def fit_and_print(metric, distance_indices, feature_M):
    fitted_dbscan_dict = {}
    best_cnt = 0
    best_dist = 0
    
    for distance_index in distance_indices:
        dbscan_params = {
            'algorithm':'auto',
            'metric':metric,
            'eps':distance_index,
            'min_samples':2,
        }
        
        if metric in ['cosine']: dbscan_params['algorithm'] = 'brute'
        fitted_dbscan = DBSCAN(**dbscan_params).fit(feature_M)
        clusters = pd.Series(fitted_dbscan.labels_) 
        cnt = len(clusters.value_counts())
        print 'distance: ', distance_index, ' cnt: ',cnt
        print clusters.value_counts()
        print 'count clusters: ', cnt, 'large clusters: ', count_of_large_clusters(clusters)
        print '-' * 50
        print
        
        fitted_dbscan_dict[str(distance_index)] = (fitted_dbscan, cnt)

        if cnt > best_cnt:
            best_cnt = cnt
            best_dist = distance_index
    
    print 'BEST number of clusters: ',best_cnt, ' eps: ',best_dist
    return fitted_dbscan_dict


def grid_search(metrics, feature_M ,range_of_eps_distances=[0.06,0.09, 0.2, 0.4, 0.6, 0.8, 1]):
    dict_dict_of_dbcans = {}
    for metric in metrics:
        # try:
            print metric.upper()
            print '=' * 50
            print
            dict_dict_of_dbcans[metric] = fit_and_print(metric,range_of_eps_distances, feature_M)
        # except:
            print 'failed @ metric -- ', metric
            continue
    return dict_dict_of_dbcans

