#packages
import numpy as np
from scipy import linalg as ls 
import networkx as nx 
from functools import reduce
from tqdm import tqdm

#########################################################################

#Complete graph Hamiltonian
def H_CG(N,gamma,w,kappa):
    #Hamiltonian matrix
    ket_s = (1.0/np.sqrt(N))*np.ones((N,1))
    H = -gamma*N*np.dot(ket_s,np.conjugate(ket_s).transpose())
        
    # Ensure H can handle complex numbers
    H = H.astype(complex)

    H[w,w] = H[w,w] - 1 - 1j*kappa
    
    return H   

#########################################################################

#calculating the overlap numerically
def overlap_CG_numerics(N,gamma,w,kappa):

    overlap_right0 = np.zeros(len(gamma),dtype=np.complex_)
    overlap_left0 = np.zeros(len(gamma),dtype=np.complex_)

    overlap_right1 = np.zeros(len(gamma),dtype=np.complex_)
    overlap_left1 = np.zeros(len(gamma),dtype=np.complex_)

    overlap_right2 = np.zeros(len(gamma),dtype=np.complex_)
    overlap_left2 = np.zeros(len(gamma),dtype=np.complex_)

    overlap_right3 = np.zeros(len(gamma),dtype=np.complex_)
    overlap_left3 = np.zeros(len(gamma),dtype=np.complex_)

    #calculate the overlap
    psi_0 = (1.0/np.sqrt(N))*np.ones((N,1))

    #target state
    ket_w = np.zeros((N,1))
    ket_w[w] = 1

    for i in tqdm(range(len(gamma))):
        
        #Hamiltonian matrix
        H = H_CG(N,gamma[i],w,kappa)
        
        #get the eigenvalue and right eigenstates
        eigval_right,eigvec_right = np.linalg.eig(H)

        # Sort the eigenvalues and eigenvectors
        sorted_indices = np.argsort(np.real(eigval_right))  # Get indices for sorting eigenvalues
        eigval_right_sorted = eigval_right[sorted_indices]  # Sort eigenvalues
        eigvec_right_sorted = eigvec_right[:,sorted_indices]  # Reorder eigenvectors accordingly    

        #get the eigenvalue and left eigenstates
        eigval_left,eigvec_left = np.linalg.eig(np.transpose(H))

        #Sort the eigenvalues and eigenvectors
        sorted_indices = np.argsort(np.real(eigval_left))  # Get indices for sorting eigenvalues
        eigval_left_sorted = eigval_left[sorted_indices]  # Sort eigenvalues
        eigvec_left_sorted = eigvec_left[:, sorted_indices]  # Reorder eigenvectors accordingly       

        overlap_right0[i] = np.vdot(psi_0,eigvec_right_sorted[:,0])
        overlap_left0[i] = np.vdot(eigvec_left_sorted[:,0],psi_0)

        overlap_right1[i] = np.vdot(psi_0,eigvec_right_sorted[:,1])
        overlap_left1[i] = np.vdot(eigvec_left_sorted[:,1],psi_0)

        overlap_right2[i] = np.vdot(ket_w,eigvec_right_sorted[:,0])
        overlap_left2[i] = np.vdot(eigvec_left_sorted[:,0],ket_w)

        overlap_right3[i] = np.vdot(ket_w,eigvec_right_sorted[:,1])
        overlap_left3[i] = np.vdot(eigvec_left_sorted[:,1],ket_w)  


    return np.abs(np.multiply(overlap_left0,overlap_right0)), np.abs(np.multiply(overlap_left1,overlap_right1)), np.abs(np.multiply(overlap_left2,overlap_right2)), np.abs(np.multiply(overlap_left3,overlap_right3))

#########################################################################

#theoretical eigenvalues in the two dimensional basis
def lambda_pm(N,gamma,kappa):

    lambda_plus  = -(gamma*N + 1 + 1.0j*kappa)/2  + np.sqrt(((gamma*N + 1 + 1.0j*kappa)/2)**2 - (gamma*N - gamma)*(1 + 1.0j*kappa))
    lambda_minus = -(gamma*N + 1 + 1.0j*kappa)/2  - np.sqrt(((gamma*N + 1 + 1.0j*kappa)/2)**2 - (gamma*N - gamma)*(1 + 1.0j*kappa))

    return lambda_plus,lambda_minus

#theoretical eigenvectors in the two dimensional basis
def ket_lambda(N,gamma,kappa):

    lambda_plus,lambda_minus = lambda_pm(N,gamma,kappa)

    a_plus = (- lambda_plus - gamma*(N -1))/(gamma*np.sqrt(N-1))
    b_plus = 1
    N_plus = 1/np.sqrt(np.abs(a_plus)**2 + 1)
    ket_lambda_plus = N_plus*np.array([a_plus,b_plus])

    a_minus = (- lambda_minus - gamma*(N -1))/(gamma*np.sqrt(N-1))
    b_minus = 1
    N_minus = 1/np.sqrt(np.abs(a_minus)**2 + 1)
    ket_lambda_minus = N_minus*np.array([a_minus,b_minus])

    return ket_lambda_plus, ket_lambda_minus


#theoretical calculation of overlaps in the two dimensional basis
def overlap_CG_theory(N,gamma,w,kappa):

    ket_s = np.array([1.0/np.sqrt(N) ,np.sqrt((N-1)/N)])
    ket_w = np.array([1,0])

    overlap_0 = np.zeros(len(gamma),dtype=np.complex128)
    overlap_1 = np.zeros(len(gamma),dtype=np.complex128)
    overlap_2 = np.zeros(len(gamma),dtype=np.complex128)
    overlap_3 = np.zeros(len(gamma),dtype=np.complex128)

    for i in range(len(gamma)):
        ket_lambda_plus, ket_lambda_minus = ket_lambda(N,gamma[i],kappa)
        overlap_0[i] = np.abs(np.vdot(ket_lambda_minus,ket_s)*np.vdot(ket_s, ket_lambda_minus))
        overlap_1[i] = np.abs(np.vdot(ket_lambda_plus,ket_s)*np.vdot(ket_s, ket_lambda_plus))
        overlap_2[i] = np.abs(np.vdot(ket_lambda_minus,ket_w)*np.vdot(ket_w, ket_lambda_minus))
        overlap_3[i] = np.abs(np.vdot(ket_lambda_plus,ket_w)*np.vdot(ket_w, ket_lambda_plus))
               
    return overlap_0,overlap_1,overlap_2,overlap_3

#########################################################################
#theoretical calculation of survival probability in the two dimensional basis
def overlap_CG_theory_surv(N,gamma,w,kappa):

    ket_s = np.array([1.0/np.sqrt(N) ,np.sqrt((N-1)/N)])
    ket_w = np.array([1,0])

    overlap_0 = np.zeros(len(gamma),dtype=np.complex_)
    overlap_1 = np.zeros(len(gamma),dtype=np.complex_)
    overlap_2 = np.zeros(len(gamma),dtype=np.complex_)
    overlap_3 = np.zeros(len(gamma),dtype=np.complex_)

    for i in tqdm(range(len(gamma))):
        ket_lambda_plus, ket_lambda_minus = ket_lambda(N,gamma[i],kappa)
        overlap_0[i] = np.vdot(ket_lambda_minus,ket_s)
        overlap_1[i] = np.vdot(ket_lambda_plus,ket_s)
        overlap_2[i] = np.vdot(ket_lambda_minus,ket_w)
        overlap_3[i] = np.vdot(ket_lambda_plus,ket_w)
               
    return overlap_0,overlap_1,overlap_2,overlap_3




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

def overlap_hypercube(dim,gamma, w, kappa):
    
    overlap_right0 = np.zeros(len(gamma),dtype=np.complex_)
    overlap_left0 = np.zeros(len(gamma),dtype=np.complex_)

    overlap_right1 = np.zeros(len(gamma),dtype=np.complex_)
    overlap_left1 = np.zeros(len(gamma),dtype=np.complex_)

    overlap_right2 = np.zeros(len(gamma),dtype=np.complex_)
    overlap_left2 = np.zeros(len(gamma),dtype=np.complex_)

    overlap_right3 = np.zeros(len(gamma),dtype=np.complex_)
    overlap_left3 = np.zeros(len(gamma),dtype=np.complex_)
   
    #Initial state
    psi_0 = (1.0/np.sqrt(2**dim))*np.ones((2**dim,1))

    #target state
    ket_w = np.zeros((2**dim,1))
    ket_w[w] = 1

    #computing overlap
    for i in tqdm(range(len(gamma))):
        
        #Hamiltonian matrix
        H = hypercube_hamiltonian_networkx(dim,gamma[i],w,kappa)
        
        #get the eigenvalue and right eigenstates
        eigval_r,eigvec_r = np.linalg.eig(H)

        #get the eigenvalue and right eigenstates
        eigval_l,eigvec_l = np.linalg.eig(np.transpose(H))        

        # Sort the eigenvalues and eigenvectors
        sorted_indices = np.argsort(np.real(eigval_r))  # Get indices for sorting eigenvalues
        eigval_r_sorted = eigval_r[sorted_indices]  # Sort eigenvalues
        eigvec_r_sorted = eigvec_r[:, sorted_indices]  # Reorder eigenvectors accordingly


        # Sort the eigenvalues and eigenvectors
        sorted_indices = np.argsort(np.real(eigval_l))  # Get indices for sorting eigenvalues
        eigval_l_sorted = eigval_l[sorted_indices]  # Sort eigenvalues
        eigvec_l_sorted = eigvec_l[:, sorted_indices]  # Reorder eigenvectors accordingly

        overlap_right0[i] = np.vdot(psi_0,eigvec_r_sorted[:,0])
        overlap_left0[i] = np.vdot(eigvec_l_sorted[:,0],psi_0)

        overlap_right1[i] = np.vdot(psi_0,eigvec_r_sorted[:,1])
        overlap_left1[i] = np.vdot(eigvec_l_sorted[:,1],psi_0)

        overlap_right2[i] = np.vdot(ket_w,eigvec_r_sorted[:,0])
        overlap_left2[i] = np.vdot(eigvec_l_sorted[:,0],ket_w)

        overlap_right3[i] = np.vdot(ket_w,eigvec_r_sorted[:,1])
        overlap_left3[i] = np.vdot(eigvec_l_sorted[:,1],ket_w)  

    
    return np.abs(np.multiply(overlap_left0,overlap_right0)), np.abs(np.multiply(overlap_left1,overlap_right1)), np.abs(np.multiply(overlap_left2,overlap_right2)), np.abs(np.multiply(overlap_left3,overlap_right3))



