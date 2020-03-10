from scipy.sparse.linalg import lsqr, lsmr, cg, svds
from scipy.sparse import coo_matrix, block_diag
import sys
sys.path.append("S3DGLPy")
from S3DGLPy.PolyMesh import *
from Util.util import *
import time
from numpy.linalg import lstsq
from scipy.sparse.linalg import spsolve
import scipy.io as sio
from numpy.linalg import qr
from scipy import linalg

if __name__ == "__main__":
    file_name = r"./data/out.obj"
    polymesh = PolyMesh()
    polymesh.loadObjFile(file_name)
    v, f = loadObj(file_name)
    L = load_pickle_file(r"./data/L.pkl")
    l = block_diag((L, L, L))
    # q, r = linalg.qr(l.todense())
    # print(q.shape)
    delta = load_pickle_file(r"./data/delta.pkl")
    gt = np.hstack((polymesh.VPos[:, 0], polymesh.VPos[:, 1], polymesh.VPos[:, 2]))
    d = np.hstack((delta[:, 0], delta[:, 1], delta[:, 2]))
    print("l shape is {}, d shape is {}".format(l.shape, d.shape))
    start = time.time()
    ans = lsqr(l, d, show=False, x0=np.array(v, dtype=np.float32).T.reshape(-1))
    print(ans[0].reshape(3, -1).T.shape)
    print("block error is {}".format(np.abs(l.dot(gt) - d).mean()))
    # ans1 = lsmr(l, d, show=True, atol=1.00e-012, btol=1.00e-06)
    # ans1 = cg(l, d, x0=np.array(v).T.reshape(-1))

    # u, s, vt = svds(l)
    # print(s)
    # print(vt.shape)
    # sio.savemat('L.mat', {'L': l})
    # sio.savemat('d.mat', {'delta': d})
    # ans = sio.loadmat('ans.mat')
    # ans_numpy = lstsq(l.todense(), d)
    # for i in range(3):
    #     polymesh.VPos[:, i] = lsqr(L, delta[:, i], show=False)[0]
    #     print(np.abs(L.dot(polymesh.VPos[:, i])-delta[:, i]).mean())
    end = time.time()
    polymesh.VPos = ans[0].reshape(3, -1).T
    print("it takes {}s".format(end - start))
    polymesh.saveObjFile(r"./data/out1.obj")