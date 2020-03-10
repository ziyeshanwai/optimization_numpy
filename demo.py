import numpy as np
from numpy.linalg import lstsq
from scipy import optimize
import scipy.io as sio


def svdsolve(a, b):
    u, s, v = np.linalg.svd(a)
    print("u is {}".format(u.shape))
    print("s is {}".format(s.shape))
    print("v is {}".format(v.shape))
    print(v)
    c = np.dot(u.T, b)
    w = np.linalg.solve(np.diag(s), c)
    x = np.dot(v.T, w)
    return x

def func(x, a, b):
    loss = a.dot(x)-b
    return loss


if __name__ == "__main__":
    """
    method 1 direct solve
    """
    m = 1000
    n = 600
    aa = np.random.randint(low=1, high=20, size=(m, n), dtype=np.int32)
    ax = np.random.randint(low=1, high=20, size=(n, 1), dtype=np.int32)
    ay = np.random.randint(low=1, high=20, size=(n, 1), dtype=np.int32)
    az = np.random.randint(low=1, high=20, size=(n, 1), dtype=np.int32)
    deltax = aa.dot(ax)
    deltay = aa.dot(ay)
    deltaz = aa.dot(az)
    delta = np.vstack((deltax, deltay, deltaz))
    print(delta.shape)
    A = np.zeros((3*m, 3*n), dtype=np.float32)
    A[0:m, 0:n] = aa
    A[m:2*m, n:2*n] = aa
    A[2*m:3*m, 2*n:3*n] = aa
    ans = lstsq(A, delta)
    gt_ans = np.vstack((ax, ay, az))
    print(np.sqrt(np.square(ans[0]-gt_ans).mean()))

    """
    method 2 iteration method solve
    """
    optimize.minimize(fun=func, x0=)


