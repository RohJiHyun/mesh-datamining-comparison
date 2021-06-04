# PREPROCESS 
# default mesh data(V, F) => Feature
#
#




from posixpath import abspath, basename
import _make_feature as feature
import igl 
import glob
import argparse
import os 
import cfilesystem
import numpy as np 
import queue

import json 


parser = argparse.ArgumentParser("artgs")
parser.add_argument("--path", required=True) # data path
parser.add_argument("--save", default="./processed") # save path 
parser.add_argument("--ref", required=True) # ref path
args = parser.parse_args()



emotion_category = [
                'bareteeth', 'cheeks_in', 'eyebrow', 'high_smile',
                'lips_back', 'lips_up', 'mouth_down', 'mouth_extreme',
                'mouth_middle','mouth_open','mouth_side','mouth_up'
                ]

allowed_file_ext = ['.ply', '.obj']
numpy_save_ext = '.npy'

def preprocess(child_path, ref_model, root_from, root_to):
    """
    """
    V, _ = igl.read_triangle_mesh(child_path)
    tmp_feature_model = feature.Model(ref_model)
    tmp_feature_model.set_data(V, ref_model.F)
    mesh_feature = tmp_feature_model.compute_feature_vector()

    child_path = cfilesystem.to_abs(child_path)
    new_child_path = cfilesystem.exchange_root_path(child_path, root_from, root_to)
    new_path, _ = os.path.splitext(new_child_path)

    new_path_root = os.path.dirname(new_path)
    if not os.path.exists(new_path_root):
        os.makedirs(new_path_root)

    name = new_path + numpy_save_ext
    np.save(name, mesh_feature)


    return True


def postprocess():
    pass



def add_category(path, dict_data, category=[], load_path="", save_path=""):
    """

    """
    name = os.path.basename(path)
    if  name in category:
        if name not in dict_data.keys():
            dict_data[name]=[]
        child_list = glob.glob(os.path.join(path,"*"))
        for i in range(len(child_list)):
            tmp_name = cfilesystem.exchange_root_path(child_list[i], load_path, save_path)
            tmp_name, _ = os.path.splitext(tmp_name)
            child_list[i] = tmp_name + numpy_save_ext
            
        dict_data[name].extend(child_list)
    

def launch(load_path, save_path, ref_path):
    def reference_model_init(ref_path):
        if not os.path.exists(ref_path):
            raise FileNotFoundError("reference_path no exists.")
            
        ref = feature.Model()
        V, F = igl.read_triangle_mesh(ref_path)
        ref.set_data(V, F)
        return ref 

    

    abs_path = os.path.abspath(load_path)
    ref = reference_model_init(ref_path) # ref init
    
    dict_data = dict()

    if not os.path.exists(abs_path):
        raise FileNotFoundError()
    



    stack = [abs_path]
    print("start")
    while len(stack) > 0 :
        picked_path = stack.pop()
        if os.path.isdir(picked_path):
            child_list = glob.glob(os.path.join(picked_path, "*"))
            stack.extend(child_list)
            add_category(picked_path, dict_data, category=emotion_category, load_path=load_path, save_path=save_path)

        elif os.path.isfile(picked_path):
            if cfilesystem.check_allowed_ext(picked_path, allowed_ext=allowed_file_ext):
                preprocess(picked_path, ref, load_path, save_path)
    print("end")

    

    # save Category data
    with open(os.path.join(save_path, "index.json"), 'w') as f:
        json.dump(dict_data, f, indent=4)
    


        

    






if __name__ == "__main__":
    launch(load_path = args.path, save_path=args.save, ref_path=args.ref)