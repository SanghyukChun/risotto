import numpy as np
from scipy.sparse import csr_matrix

from methods.als import ALS


if __name__ == '__main__':
    n_users, n_items = 943, 1682
    R = np.zeros([n_users, n_items])
    with open('./ml-100k/u.data', 'r') as f:
        for line in f.readlines():
            u, i, r, _ = map(lambda x: int(x), line.split())
            R[u-1, i-1] = r
    R = csr_matrix(R)
    solver = ALS(R, 50, reg=0.05, max_iter=100)
    solver.train()
    solver.rmse(R)
