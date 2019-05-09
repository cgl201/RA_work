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


class State_space_model:
    def __init__(self, output_dim, initial_state):
        self.output_dim = output_dim
        self.state_dim = len(initial_state)
        self.initial_state = initial_state

    def create_measure(self, measurement_para, measurement_para_position, measurement_one_position):
        self.measure_para = measurement_para
        self.measure_para_position = measurement_para_position
        self.measure_one_position = measurement_one_position
        self.H = self.gen_mat(self.measure_para, self.measure_para_position, self.measure_one_position, True)

    def create_transition(self, transition_para, transition_para_position, transition_one_position):
        self.trans_para = transition_para
        self.trans_para_position = transition_para_position
        self.trans_one_position = transition_one_position
        self.F = self.gen_mat(self.trans_para, self.trans_para_position, self.trans_one_position, False)

    def gen_mat(self, para, para_position, one_position, measure_flag):
        self.row = [i[0] for i in para_position] + [j[0] for j in one_position]
        self.col = [i[1] for i in para_position] + [j[1] for j in one_position]
        self.para = para + [1] * len(one_position)
        if measure_flag:
            return coo_matrix((self.para, (self.row, self.col)), shape=(self.output_dim, self.state_dim)).toarray()
        else:
            return coo_matrix((self.para, (self.row, self.col)), shape=(self.state_dim, self.state_dim)).toarray()

    def urand(self, sigma):
        self.x = np.random.normal(0, sigma)
        return self.x

    def create_measure_noise(self, measure_noise_para, measure_noise_position):
        self.measure_noise_para = measure_noise_para
        self.measure_noise_position = measure_noise_position

    def create_trans_noise(self, trans_noise_para, trans_noise_position):
        self.trans_noise_para = trans_noise_para
        self.trans_noise_position = trans_noise_position

    def gen_noise(self, para, para_position, noise_flag):
        self.row1 = [i for i in para_position]
        self.col1 = [0] * len(para)
        if noise_flag:
            return coo_matrix((urand(np.array(para)), (self.row1, self.col1)), shape=(self.output_dim, 1)).toarray()
        else:
            return coo_matrix((urand(np.array(para)), (self.row1, self.col1)), shape=(self.state_dim, 1)).toarray()

    def predict(self, time_periods):
        self.time = time_periods
        self.state = self.initial_state
        self.g = []
        self.u = []
        for i in range(self.time):
            self.state = np.matmul(self.F, self.state) + self.gen_noise(self.trans_noise_para, self.trans_noise_position, False)
            self.y = np.matmul(self.H, self.state) + self.gen_noise(self.measure_noise_para, self.measure_noise_position, True)
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


model = State_space_model(2, np.zeros((6, 1)))
model.create_transition([1.4386, -0.5174], [(1, 1), (1, 2)], [(0, 0), (0, 4), (2, 1), (3, 2), (4, 4), (5, 5)])
model.create_measure([-0.3368, -0.1635, -0.0720], [(1, 1), (1, 2), (1, 3)], [(0, 0), (0, 1), (1, 5)])

model.create_measure_noise([0.0003], [1])
model.create_trans_noise([0.0049, 0.0067, 0.0003, 0.0015], [0, 1, 4, 5])

model.simulation(36, 100)
print(model.data_u)

model.data_u.plot(legend=False)
plt.show()
