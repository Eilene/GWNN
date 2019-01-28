import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse
import sys
import math
import warnings
warnings.filterwarnings("ignore")

def adj_matrix():
    names = [ 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format("cora", names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects = pkl.load(f, encoding='latin1')
            else:
                objects = pkl.load(f)
    graph = objects
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj

def laplacian(W, normalized=False):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        # d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

def fourier(dataset,L, algo='eigh', k=100):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""
    # print "eigen decomposition:"
    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]
    # if(dataset == "pubmed"):
    #     # print "loading pubmed U"
    #     rfile = open("data/pubmed_U.pkl")
    #     lamb, U = pkl.load(rfile)
    #     rfile.close()
    # else:
    if algo is 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')
    # print "end"
    # wfile = open("data/pubmed_U.pkl","w")
    # pkl.dump([lamb,U],wfile)
    # wfile.close()
    # print "pkl U end"
    return lamb, U

def weight_wavelet(s,lamb,U):
    s = s
    for i in range(len(lamb)):
        lamb[i] = math.pow(math.e,-lamb[i]*s)

    Weight = np.dot(np.dot(U,np.diag(lamb)),np.transpose(U))

    return Weight

def weight_wavelet_inverse(s,lamb,U):
    s = s
    for i in range(len(lamb)):
        lamb[i] = math.pow(math.e, lamb[i] * s)

    Weight = np.dot(np.dot(U, np.diag(lamb)), np.transpose(U))

    return Weight





