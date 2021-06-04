import sklearn as sk 

import numpy as np 
from sklearn.decomposition import PCA 
from sklearn.decomposition import IncrementalPCA
import cfilesystem as cf
import argparse

import json
import os 
import random







parser = argparse.ArgumentParser("artgs")
parser.add_argument("--path", required=True) # data path
parser.add_argument("--save", default="./reduced") # save path 
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

random.seed(env['seed'])


emotion_category = [
                'bareteeth', 'cheeks_in', 'eyebrow', 'high_smile',
                'lips_back', 'lips_up', 'mouth_down', 'mouth_extreme',
                'mouth_middle','mouth_open','mouth_side','mouth_up'
                ]
def save_data(dataset, from_path, save_path, conf_file=None):
    for path, data in dataset:
        new_save_path = cf.exchange_root_path(path, from_path, save_path)
        dirname = os.path.dirname(new_save_path)
         
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        np.save(new_save_path, data)


    if conf_file != None : #conf file is dictionary json
        for key in conf_file:
            for idx in range(len(conf_file[key])):
                conf_file[key][idx] = cf.exchange_root_path(conf_file[key][idx], from_path, save_path)

        with open(os.path.join(save_path, args.conf), 'w') as f:
            json.dump(conf_file, f, indent=4)
        
    

    


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
    for name in file_name_list:
        data = np.load(name)
        dataset.append([name, data])
    
    return dataset, conf




def launch_PCA(dataset, batch_size, ratio):
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

    def PCA_process(data_chunk, batch_size, ratio_or_n_components=0.8):
        orig_component_size = data_chunk.shape[-1] 
        if isinstance(ratio_or_n_components, float):
            size = orig_component_size * ratio
        elif isinstance(ratio_or_n_components, int) : # or ... integert
            size = ratio_or_n_components
        else : 
            size = orig_component_size



        total_batch = size // batch_size + (1 if size % batch_size > 0 else 0)
        print("total datachunk : ", size)
        print("batch size : ", batch_size)
        print("total batch : ", total_batch)

        inc_pca = IncrementalPCA(n_components=int(size), batch_size = batch_size)
        X_reduced = inc_pca.fit_transform(data_chunk)
        # for i in range(total_batch):
        #     start = batch_size*(i)
        #     end = start + batch_size*(i+1)
        #     batch_x = data_chunk[start:end]
        #     print(start, " ", end)
        #     inc_pca.partial_fit(batch_x)
        #     print("test")
        
        # X_reduced = inc_pca.transform(data_chunk)

        return X_reduced
    
    def postprocess(numpy_chunk, dataset):
        """
            numpy -> dataset : (name, data(M, 1))
        """
        for i in range(len(numpy_chunk)):
            dataset[i][-1] = numpy_chunk[i, np.newaxis].T

        return dataset
    data_chunk = preprocess(dataset)
    reduced_dim_data_chunk = PCA_process(data_chunk, batch_size, ratio)
    dataset = postprocess(reduced_dim_data_chunk, dataset)
    
    return dataset


if __name__ == "__main__":
    print("data load")
    data, conf_data = load_data(env['path'], env['conf'])
    print("data process")
    transformed_data = launch_PCA(data, env['batch'], env['ratio'])
    print("data save")
    save_data(transformed_data, env['path'], env['save'], conf_data)
