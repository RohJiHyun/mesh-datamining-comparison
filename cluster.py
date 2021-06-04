




from sklearn.cluster import KMeans, DBSCAN
import os, json, random
import numpy as np
import argparse

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




def load_data(root_path, conf_file):
    # preproc
    with open(os.path.join(root_path, conf_file), 'r') as f:
        conf = json.load(f)
    
    file_name_list = []
    for category in emotion_category:
        file_name_list.extend(conf[category])

    random.shuffle(file_name_list)


    #load
    dataset = []
    for name in file_name_list[:1000]:
        data = np.load(name)
        dataset.append([name, data])
    
    return dataset

def preprocess(dataset):
        """
            dataset : (name, data(M, 1)) -> numpy (N, M)
        """
        data_list = []
        for _, data in dataset:
            data_list.append(data.T) # M, 1 -> 1, M


        reval = np.array(data_list) # N, M
        assert reval.shape[0] == len(dataset) and reval.shape[-1] == dataset[0][-1].shape[0], "shape mismatching"

        reval = np.squeeze(reval)
        print("preproc shape", reval.shape)
        return reval
############
### DRAW ###
############
def plotting(data, algorithmed_label, orig_label):
    import matplotlib.pyplot as plt 
    

    



def lauch_K_mean_cluster(dataset):
    print("start K_MEAN")
    cat_num = len(emotion_category)
    cluster = KMeans(cat_num)
    label = cluster.fit(dataset)
    print(label.labels_)
    print("end K_MEAN")
    return label
    



def launch_DBSCAN(dataset):
    print("start DBSCAN")
    category_num = len(emotion_category)

    dbscan_obj = DBSCAN(eps=50, min_samples=category_num)
    labels = dbscan_obj.fit(dataset)
    print(labels.labels_)
    print("end DBSCAN")
    return labels
def launch(dataset):
    datachunk = preprocess(dataset)
    label = launch_DBSCAN(datachunk)
    label = lauch_K_mean_cluster(datachunk)


if __name__ == "__main__":
    dataset = load_data(env['path'], env['conf'])
    launch(dataset)