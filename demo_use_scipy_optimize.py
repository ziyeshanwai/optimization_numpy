import scipy.optimize
import sys
sys.path.append("S3DGLPy")
from S3DGLPy.PolyMesh import *
from Util.util import *
import time
from scipy.optimize import least_squares

def func_res(x, L, b):
    X = x.reshape(-1, 3)
    loss = np.sqrt(np.sum(np.square(L.dot(X)-b), axis=0)).mean()
    return loss


if __name__ == "__main__":
    file_name = r"./data/in.obj"
    polymesh = PolyMesh()
    polymesh.loadObjFile(file_name)
    v, f = loadObj(file_name)
    L = load_pickle_file(r"./data/L.pkl")
    # L = L.todense()
    delta = load_pickle_file(r"./data/delta.pkl")
    start = time.time()
    x = polymesh.VPos
    # res = scipy.optimize.minimize(func_res, x0=x.reshape(-1), args=(L, delta), method='CG', jac=False,
    #                               options={"gtol": 1e-7, "maxiter": 200, "disp":True, "eps": 1e-3})
    res = least_squares(func_res, x0=x.reshape(-1), args=(L, delta), method='trf')
    print(res.success)
    print(res.x)
    print(res.cost)
    polymesh.VPos = res.x.reshape(-1, 3)
    # for i in range(3):
    #     polymesh.VPos[:, i] = lsqr(L, delta[:, i])[0]
    end = time.time()
    print("it takes {}s".format(end - start))
    polymesh.saveObjFile(r"./data/out.obj")