import scipy.linalg
import sys
sys.path.append("S3DGLPy")
from S3DGLPy.PolyMesh import *
from Util.util import *
import time

def func_res(x, L, b):
    X = x.reshape(-1, 3)
    return np.sqrt(np.sum(np.square(L.dot(X)-b), axis=1)).mean()


if __name__ == "__main__":
    file_name = r"./data/in.obj"
    polymesh = PolyMesh()
    polymesh.loadObjFile(file_name)
    v, f = loadObj(file_name)
    L = load_pickle_file(r"./data/L.pkl")
    L = L.todense()
    delta = load_pickle_file(r"./data/delta.pkl")
    start = time.time()
    for i in range(3):
        polymesh.VPos[:, i] = scipy.linalg.lstsq(L, delta[:, i])[0]
    end = time.time()
    print("it takes {}s".format(end - start))
    polymesh.saveObjFile(r"./data/out.obj")