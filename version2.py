# -*- coding: utf-8 -*-
# Time: 2/23/2019 7:50 PM
# Author: Guanlin Chen
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.sparse import coo_matrix


def urand(sigma):
    x = np.random.normal(0, sigma)
    return x

def zeromat(nr,nc): 
    return(np.matrix(np.zeros((nr,nc))))

class parameter:
    def __init__(self, para, para_position, one_position, type_flag, model):
        self.para = para
        self.para_position = para_position
        self.one_position = one_position
        mat_dict = {'measure': 'H',
                    'transit': 'F',
                    'measure_noise': 'e_H',
                    'transit_noise': 'e_F'}
        self.mat_flag = mat_dict.get(type_flag)
        self.model = model
    def matrix(self):
        if self.one_position:
            row = [i[0] for i in self.para_position] + [j[0] for j in self.one_position]
            col = [i[1] for i in self.para_position] + [j[1] for j in self.one_position]
            mat_val = self.para + [1] * len(self.one_position)
        else:
            row = [i for i in self.para_position]
            col = [0] * len(self.para)
            mat_val = urand(np.array(self.para))
            
        mat_shape = {'H': (self.model.output_dim, self.model.state_dim),
                     'F': (self.model.state_dim, self.model.state_dim),
                     'e_H': (self.model.output_dim, 1),
                     'e_F': (self.model.state_dim, 1)}
        
        return coo_matrix((mat_val, (row,col)), shape=mat_shape.get(self.mat_flag)).toarray()


class State_space_model:
    def __init__(self, output_dim, initial_state):
        self.output_dim = output_dim
        self.state_dim = len(initial_state)
        self.initial_state = initial_state
        param_shape = {'H': (self.output_dim, self.state_dim),
                     'F': (self.state_dim, self.state_dim)}
        for parname in param_shape:
            setattr(self,parname,zeromat(*param_shape.get(parname)))
    
    def initialize(self, param_list):
        noise_dict = {'e_H','e_F'}
        for param in param_list:
            if param.mat_flag not in noise_dict:
                setattr(self,param.mat_flag,param.matrix())
            else:
                setattr(self,param.mat_flag,param)
    
    def predict(self, time_periods):
        self.time = time_periods
        self.state = self.initial_state
        self.g = []
        self.u = []
        for i in range(self.time):
            self.state = np.matmul(self.F, self.state) + self.e_F.matrix()
            self.y = np.matmul(self.H, self.state) + self.e_H.matrix()
            self.g.append(self.y[0][0])
            self.u.append(self.y[1][0])

    def simulation(self, time_periods, N):
        self.N = N
        self.time = time_periods
        self.data_g = pd.DataFrame()
        self.data_u = pd.DataFrame()

        for n in range(N):
            self.predict(self.time)
            self.data_g[n] = self.g
            self.data_u[n] = self.u


model = State_space_model(2, np.zeros((6, 1))) # output_dim=2,state_dim=6

measure_param = parameter([-0.3368, -0.1635, -0.0720], [(1, 1), (1, 2), (1, 3)], [(0, 0), (0, 1), (1, 5)],'measure',model)
transit_param = parameter([1.4386, -0.5174], [(1, 1), (1, 2)], [(0, 0), (0, 4), (2, 1), (3, 2), (4, 4), (5, 5)],'transit',model)
measure_noise_param = parameter([0.0003], [1], [], 'measure_noise', model)
transit_noise_param = parameter([0.0049, 0.0067, 0.0003, 0.0015], [0, 1, 4, 5], [], 'transit_noise', model)

model.initialize([measure_param,transit_param,measure_noise_param,transit_noise_param])

model.simulation(36, 100)
print(model.data_u)

model.data_u.plot(legend=False)
plt.show()
