import numpy as np 
import igl 
import numpy as np 

import scipy.sparse as sp 


class Model():
    """
        class Model

        compute feature vector
    """
    def __var_init(self):
        self.is_ref = True
        self.F = None 
        self.extended_F = None
        self.V = None
        self.extended_V = None
        self.G = None
        self.data = None
        self.feature_vec = None 

    def __init__(self, ref_obj=None):
        self.__var_init()


        # self.ref_obj = ref_obj
        if ref_obj :
            self.is_ref = False
            self.ref_obj = ref_obj
        




    def set_data(self, V, F):
        self.V = V 
        self.F = F
        
        self._precompute()
        if self.is_ref : 
            self.construct_G()
        else : 
            self.G = self.ref_obj.G
        

    def data_flatten(self):
        self.data = self.extended_V.T.reshape(-1, 1)
        return self.data

    def _precompute(self):
        
        edge1 = self.V[self.F[:, 1]] - self.V[self.F[:, 0]]
        edge2 = self.V[self.F[:, 2]] - self.V[self.F[:, 0]]
        face_n = np.cross(edge1, edge2)
        norm = np.linalg.norm(face_n, axis=1)
        
        face_n /= norm[:, np.newaxis]
        v4 = face_n + self.V[self.F[:, 0]]

        extend_list = np.arange(self.V.shape[0], self.V.shape[0]+v4.shape[0]).reshape(-1,1) # n ~ v4_size


        self.extended_V = np.concatenate([self.V, v4])
        self.extended_F = np.concatenate([self.F, extend_list], axis=-1)




    def construct_G(self):
        print("G constructions start...")
        self.n = self.extended_V.shape[0]
        self.m = self.extended_F.shape[0]
        
        v14 = self.extended_V[self.extended_F[:, 0]] - self.extended_V[self.extended_F[:, 3]] # [N, 3]
        v24 = self.extended_V[self.extended_F[:, 1]] - self.extended_V[self.extended_F[:, 3]] # [N, 3]
        v34 = self.extended_V[self.extended_F[:, 2]] - self.extended_V[self.extended_F[:, 3]] # [N, 3]


        concated_po = np.stack([v14, v24, v34], axis=-1) #[v1-v4 v2-v4 v3-v4] it means already transposed. 

        # Transposed and inversed concated_po := [-- v10' --]
        #                                        [-- v20' --]
        #                                        [-- v30' --] 
        concated_po = np.array(list(map(lambda x : np.linalg.inv(x), concated_po)))
        concated_po = np.transpose(concated_po, [0,2,1])

        # for i in range(concated_po.shape[0]): # Calc Inv
            
        #     inv = np.linalg.inv( concated_po[i, ...] )
        #     concated_po[i, ...] = inv.T

        # concated_po = inv((p)T)
        # 9m X 3n

        self.G = sp.csc_matrix((self.F.shape[0]*9, self.extended_V.shape[0]*3)) # for Sparse


        poi_sum = np.sum(concated_po, axis=-1) # N x 3


        for i in range(3):
            col_list = self.extended_F[:, -1].reshape(-1) + self.n*i
            row1_list = np.arange(3*self.m*i + 0, 3*self.m*(i+1) + 0, 3)
            row2_list = np.arange(3*self.m*i + 1, 3*self.m*(i+1) + 1, 3)
            row3_list = np.arange(3*self.m*i + 2, 3*self.m*(i+1) + 2, 3)
            
            # assign  (-a-b-c)
            self.G[ row1_list, col_list] = -poi_sum[:, 0]
            self.G[ row2_list, col_list] = -poi_sum[:, 1]
            self.G[ row3_list, col_list] = -poi_sum[:, 2]

            
            # assign (a b c)
            
            vertice_col_ind = self.extended_F[:, [0,1,2]] + self.n*i

            expanded_row1_list = row1_list.reshape(-1,1)
            expanded_row2_list = row2_list.reshape(-1,1)
            expanded_row3_list = row3_list.reshape(-1,1)
            self.G[ expanded_row1_list, vertice_col_ind] = concated_po[:, 0]
            self.G[ expanded_row2_list, vertice_col_ind] = concated_po[:, 1]
            self.G[ expanded_row3_list, vertice_col_ind] = concated_po[:, 2]




        print("G constructions end...")


    def compute_feature_vector(self):
        
        self.feature_vec = self.G.dot(self.data_flatten())
        return self.feature_vec



    

