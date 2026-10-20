import numpy as np
import os
import pickle

class FinancialNetwork:
    rng = np.random.default_rng(55)
    
    def __init__(self, N):
        self.N = N # 国の数
        raw_p = FinancialNetwork.rng.uniform(0, 1, N)
        self.p = raw_p / raw_p[np.argmin(raw_p)] # 最小値によって正規化
        self.C, self.A = self.calc_A(self.N)
        self.D = np.eye(self.N)
        v0 = self.calc_v0()
        self.threshold = 0.8 * v0
        self.beta = 0.5 * self.threshold
        self.default = FinancialNetwork.rng.integers(0, self.N)
        self.p_d = self.p.copy()
        self.p_d[self.default] = 0
        self.x_max = self.A @ self.p_d 
        self.x_min = self.A @ (self.p_d - self.beta)
        
    def calc_v0(self):
        return self.A @ self.D @ self.p 
        
    @staticmethod 
    def calc_A(N):
        C_bar = FinancialNetwork.rng.uniform(0, 1, (N, N))
        lam = 1.0
        L = (FinancialNetwork.rng.exponential(1/lam, size=N) + 1) * np.sum(C_bar, axis = 0)  
        C = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                C[i][j] = C_bar[i][j] / L[j]
        diag_element = 1 - np.sum(C, axis = 0)
        C_hat = np.diag(diag_element)
        A = C_hat @ np.linalg.inv(np.eye(N) - C)  
        return C, A
        
    def get_problem_parameters(self):
        keys = ['N', 'p', 'C', 'A', 'D', 'threshold', 'beta',
                'default', 'p_d', 'x_max', 'x_min']
        return {k: getattr(self, k) for k in keys}

N_list = np.arange(2, 22, 2)
n_instances = 50

for N in N_list:
    problem_parameters_list = []
    for i in range(n_instances):
        problem = FinancialNetwork(N)
        problem_parameters_list.append(problem.get_problem_parameters())
    filename = f"instances/N={N}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(problem_parameters_list, f)    