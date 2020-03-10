import numpy as np
from numpy.linalg import lstsq
from scipy import optimize
import scipy.io as sio
import time


def svdsolve(a, b):
    u, s, v = np.linalg.svd(a)
    c = np.dot(u.T, b)
    w = np.linalg.solve(np.diag(s), c)
    x = np.dot(v.T, w)
    return x

def func(x, a, b):
    loss = np.square(a.dot(x)-b).mean()  # should be scalar
    return loss


if __name__ == "__main__":
    """
    method 1 direct solve
    """
    m = 100
    n = 60
    aa = np.random.randint(low=1, high=20, size=(m, n), dtype=np.int32)
    ax = np.random.randint(low=1, high=20, size=(n, 1), dtype=np.int32)
    ay = np.random.randint(low=1, high=20, size=(n, 1), dtype=np.int32)
    az = np.random.randint(low=1, high=20, size=(n, 1), dtype=np.int32)
    deltax = aa.dot(ax)
    deltay = aa.dot(ay)
    deltaz = aa.dot(az)
    delta = np.vstack((deltax, deltay, deltaz))
    A = np.zeros((3*m, 3*n), dtype=np.float32)
    A[0:m, 0:n] = aa
    A[m:2*m, n:2*n] = aa
    A[2*m:3*m, 2*n:3*n] = aa
    print("")
    start = time.time()
    ans = lstsq(A, delta)
    gt_ans = np.vstack((ax, ay, az))
    error = np.sqrt(np.square(ans[0]-gt_ans).mean())
    end = time.time()
    print("lstsq takes {}s, error is {}".format(end - start, error))

    """
    method 2 iteration method solve
    """
    start = time.time()
    res = optimize.minimize(fun=func, x0=np.random.randint(low=1, high=20, size=(A.shape[1], 1), dtype=np.int32), args=(A, delta))
    err = (A.dot(res.x) - delta).mean()
    print("error is {}".format(err))
    error = np.sqrt(np.square(res.x - gt_ans).mean())
    end = time.time()
    print("scipy optimize takes {}s, error is {}".format(end - start, error))
