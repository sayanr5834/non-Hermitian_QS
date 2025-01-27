#packages
import numpy as np
from scipy import linalg as ls 
import networkx as nx 
from functools import reduce


#########################################################################

#Networkx hypercube hamiltonian
def hypercube_hamiltonian_networkx(dim,gamma, w, kappa):

    G = nx.hypercube_graph(dim)
    
    H = 0.5*gamma*nx.laplacian_matrix(G).toarray()

    # Ensure H can handle complex numbers
    H = H.astype(complex)
    
    #The search Hamiltonian with dissipation
    H[w,w] = H[w,w] - 1 - 1j*kappa

    return H

#########################################################################

#manually generated hypercube Hamiltonian

from functools import reduce

def pauli_x():
    #Pauli-X matrix
    return np.array([[0, 1],
                     [1, 0]])

def identity():
    #2x2 Identity matrix
    return np.eye(2)

def kronecker_sum_x(n):
    """
    Construct the adjacency matrix of an n-dimensional hypercube
    using Pauli-X matrices.
    """
    size = 2**n  # Total number of vertices
    adjacency_matrix = np.zeros((size, size))

    # Generate the adjacency matrix using the Kronecker sum
    for i in range(n):
        op = [identity()] * n
        #replace the ith element by the pauli x operator
        op[i] = pauli_x()

        #The reduce function applies the np.kron cumulatively to the elements in op
        adjacency_matrix += reduce(np.kron, op)

    return adjacency_matrix

def hypercube_hamiltonian(dim,gamma, w, kappa):
    adj_matrix = kronecker_sum_x(dim)
    deg_matrix = dim*np.identity(2**dim)

    laplacian = deg_matrix - adj_matrix

    H = 0.5*gamma*laplacian

    # Ensure H can handle complex numbers
    H = H.astype(complex)
    
    #The search Hamiltonian with dissipation
    H[w,w] = H[w,w] - 1 - 1j*kappa

    return H

#########################################################################