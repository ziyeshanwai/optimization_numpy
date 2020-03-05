from scipy.sparse.linalg import lsqr
import sys
sys.path.append("S3DGLPy")
from S3DGLPy.PolyMesh import *
from Util.util import *
import time

if __name__ == "__main__":
    file_name = r"./data/in.obj"
    polymesh = PolyMesh()
    polymesh.loadObjFile(file_name)
    v, f = loadObj(file_name)
    L = load_pickle_file(r"./data/L.pkl")
    delta = load_pickle_file(r"./data/delta.pkl")
    start = time.time()
    for i in range(3):
        polymesh.VPos[:, i] = lsqr(L, delta[:, i])[0]
    end = time.time()
    print("it takes {}s".format(end - start))
    polymesh.saveObjFile(r"./data/out.obj")