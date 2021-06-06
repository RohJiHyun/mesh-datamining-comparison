




from sklearn.cluster import KMeans, DBSCAN
import os, json, random
import numpy as np
import pandas as pd
import math
import argparse

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser("artgs")
parser.add_argument("--path", required=True) # data path
parser.add_argument("--save", default="./result") # save path 
parser.add_argument("--batch", default=100) # save path 
parser.add_argument("--ratio", default=100) # save path 
parser.add_argument("--conf", default="index.json") # save path 
parser.add_argument("--rseed", default=777) # save path 
args = parser.parse_args()


env = dict()
env['batch'] = args.batch
env['path'] = args.path
env['save'] = args.save
env['seed'] = args.rseed
env['conf'] = args.conf
env['ratio'] = args.ratio
emotion_category = [
                'bareteeth', 'cheeks_in', 'eyebrow', 'high_smile',
                'lips_back', 'lips_up', 'mouth_down', 'mouth_extreme',
                'mouth_middle','mouth_open','mouth_side','mouth_up'
                ]

emotion_category_dict = {
                'bareteeth' : 0, 'cheeks_in' : 1, 'eyebrow' :2, 'high_smile' : 3,
                'lips_back' : 4, 'lips_up' :5, 'mouth_down' : 6, 'mouth_extreme' : 7,
                'mouth_middle' : 8,'mouth_open' : 9, 'mouth_side' : 10,'mouth_up' :  11
                }



def swap_category_to_num(dataset):
    data_list = []
    label_list = []
    for _, cat, _ in dataset:
        data_list.append( emotion_category_dict[cat])
        label_list.append(cat)

    return data_list, label_list


def k_distances(X, n=None, dist_func=None):
    """Function to return array of k_distances.

    X - DataFrame matrix with observations
    n - number of neighbors that are included in returned distances (default number of attributes + 1)
    dist_func - function to count distance between observations in X (default euclidean function)
    """
    if type(X) is pd.DataFrame:
        X = X.values
    k=0
    if n == None:
        k=X.shape[1]+2
    else:
        k=n+1

    if dist_func == None:
        # euclidean distance square root of sum of squares of differences between attributes
        dist_func = lambda x, y: math.sqrt(
            np.sum(
                np.power(x-y, np.repeat(2,x.size))
            )
        )

    Distances = pd.DataFrame({
        "i": [i//10 for i in range(0, len(X)*len(X))],
        "j": [i%10 for i in range(0, len(X)*len(X))],
        "d": [dist_func(x,y) for x in X for y in X]
    })
    import matplotlib.pyplot as plt

    eps_dist = np.sort([g[1].iloc[k].d for g in iter(Distances.groupby(by="i"))])
    print(eps_dist)
    plt.hist(eps_dist,bins=30)
    plt.ylabel('n')
    plt.xlabel('Epsilon distance')
    plt.show()
    return np.sort([g[1].iloc[k].d for g in iter(Distances.groupby(by="i"))])

def load_data(root_path, conf_file):
    # preproc
    with open(os.path.join(root_path, conf_file), 'r') as f:
        conf = json.load(f)
    
    file_name_list = []
    for category in emotion_category:
        for name in conf[category]:
            file_name_list.append( (name, category) )

    random.shuffle(file_name_list)


    #load
    dataset = []
    for name, category in file_name_list[:100]:
        data = np.load(name)
        dataset.append([name, category, data])
    
    return dataset

def preprocess(dataset):
        """
            dataset : (name, data(M, 1)) -> numpy (N, M)
        """
        data_list = []
        for _,_, data in dataset:
            data_list.append(data.T) # M, 1 -> 1, M


        reval = np.array(data_list) # N, M
        assert reval.shape[0] == len(dataset) and reval.shape[-1] == dataset[0][-1].shape[0], "shape mismatching"

        reval = np.squeeze(reval)
        print("preproc shape", reval.shape)
        return reval
############
### DRAW ###
############
def plotting(transformed_data, predict_label, orig_label, label_name_list):

    def show(xs,ys, c, legend):
        dic = dict()
        for check in emotion_category:
            l = []
            for idx, name in enumerate(legend):
                if name == check:
                   l.append(idx)    
            dic[check] = l
    


        for category in emotion_category:
            idx_list = dic[category]
            print(np.take(c, idx_list))
            t = np.take(c, idx_list)*20
            print("cate", category)
            plt.scatter(np.take(xs,idx_list), np.take(ys,idx_list), c=t, label=category)
            # plt.scatter(xs[i], ys[i], c=c[i], label=legend[i])
        plt.legend()
        plt.show()

    
    transformed = transformed_data


    xs = transformed[:,0]
    ys = transformed[:,1]
    print("orig")
    print(orig_label)
    show(xs,ys,c=orig_label*100, legend=label_name_list)
    print("predict")
    show(xs,ys,c=predict_label, legend=label_name_list)
    



def lauch_K_mean_cluster(dataset):
    print("start K_MEAN")
    cat_num = len(emotion_category)
    cluster = KMeans(cat_num)
    label = cluster.fit(dataset)
    print(label.labels_)
    print("end K_MEAN")
    return label.labels_
    



def launch_DBSCAN(dataset):
    print("start DBSCAN")
    category_num = len(emotion_category)
    import math
    # dbscan_obj = DBSCAN(eps=50, min_samples=category_num)
    minPts = int(math.log(category_num)) # heuristic 
    k_dist = k_distances(dataset,n=minPts)
    check_min = 0
    idx = 0
    for i in range(1, len(k_dist - 1)):
        diff = k_dist[i] -k_dist[i-1] 
        if diff > check_min:
            check_min = diff
            idx = i
    
    print("eps", check_min)
    # dbscan_obj = DBSCAN(eps= k_dist[idx], min_samples=minPts)
    dbscan_obj = DBSCAN(eps= 80, min_samples=minPts)
    labels = dbscan_obj.fit(dataset)

    print("end DBSCAN")
    return labels.labels_
def launch(dataset):
    datachunk = preprocess(dataset)

    model = TSNE(n_components=2)
    transformed = model.fit_transform(datachunk)


    label = launch_DBSCAN(datachunk)
    print("DB SCAN PRED")
    plotting(transformed, label, *swap_category_to_num( dataset ) )
    label = lauch_K_mean_cluster(datachunk)
    plotting(transformed, label, *swap_category_to_num( dataset ) )


if __name__ == "__main__":
    dataset = load_data(env['path'], env['conf'])
    launch(dataset)