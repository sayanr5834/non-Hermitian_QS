import numpy as np
import scipy.linalg as la


# ------------------------------ Core functions -------------------------------

def H_CG(N, gamma, w, kappa):
    """
    Generate the Hamiltonian for search on the Complete Graph with a sink at target site.
    
    Args:
        N (int): Number of nodes in complete graph (system size).
        gamma (float): Prefactor scaling the QWer evolution; controls the 'hopping strength'. Optimum for CD at gamma*N = 1.
        w (int): Target node/site location; dynamics invariant under choice of w.
        kappa (float): Loss rate / strength of incoherent term in the oracle part of the Hamiltonian.
        
    Returns:
        np.ndarray: The resulting Hamiltonian in matrix form.
    """
    ket_s = np.ones((N, 1)) / np.sqrt(N) # Initial state (uniform superposition)
    H = -gamma * N * ket_s @ ket_s.T.conj() # Hamiltonian matrix of complete graph
    H = H.astype(complex)
    H[w, w] -= (1 + 1j * kappa)  # Target site
    return H.astype(complex)


def lambda_pm(N,gamma,kappa):

    """
    Compute the analytical expresion of the eigenvalues in the two dimensional subspace.

    Args:
        N (int): Number of nodes in complete graph (system size).
        gamma (float): Prefactor scaling the QWer evolution; controls the 'hopping strength'. Optimum for CD at gamma*N = 1.
        kappa (float): Loss rate / strength of incoherent term in the oracle part of the Hamiltonian.

    Returns:
        lambda_plus (float): Eigenenergy of the form a + b
        lambda_minus (float): Eigenenergy of the form a - b
    """
    

    lambda_plus  = -(gamma*N + 1 + 1.0j*kappa)/2  + np.sqrt(((gamma*N + 1 + 1.0j*kappa)/2)**2 - (gamma*N - gamma)*(1 + 1.0j*kappa))
    lambda_minus = -(gamma*N + 1 + 1.0j*kappa)/2  - np.sqrt(((gamma*N + 1 + 1.0j*kappa)/2)**2 - (gamma*N - gamma)*(1 + 1.0j*kappa))

    return lambda_plus,lambda_minus



def lambdaR(N,gamma,kappa):

    """
    Compute the analytical expresion of the right eigenvectors in the two dimensional subspace.

    Args:
        N (int): Number of nodes in complete graph (system size).
        gamma (float): Prefactor scaling the QWer evolution; controls the 'hopping strength'. Optimum for CD at gamma*N = 1.
        kappa (float): Loss rate / strength of incoherent term in the oracle part of the Hamiltonian.

    Returns:
        lambdaR_plus (np.ndarray): Right eigenvector corresponding to eigenvalue lambda_plus (column vector)
        lambdaR_minus (np.ndarray): Right eigenvector corresponding to eigenvalue lambda_minus (column vector)
    """   

    lambda_plus,lambda_minus = lambda_pm(N,gamma,kappa)

    a_plus = (- lambda_plus - gamma*(N -1))/(gamma*np.sqrt(N-1))
    b_plus = 1
   
    lambdaR_plus = np.array([a_plus,b_plus])

    a_minus = (- lambda_minus - gamma*(N -1))/(gamma*np.sqrt(N-1))
    b_minus = 1
    lambdaR_minus = np.array([a_minus,b_minus])

    return lambdaR_plus, lambdaR_minus


def lambdaL(N,gamma,kappa):

    """
    Compute the analytical expresion of the left eigenvectors in the two dimensional subspace.

    Args:
        N (int): Number of nodes in complete graph (system size).
        gamma (float): Prefactor scaling the QWer evolution; controls the 'hopping strength'. Optimum for CD at gamma*N = 1.
        kappa (float): Loss rate / strength of incoherent term in the oracle part of the Hamiltonian.

    Returns:
        lambdaL_plus (np.ndarray): Left eigenvector corresponding to eigenvalue lambda_plus (column vector)
        lambdaL_minus (np.ndarray): Left eigenvector corresponding to eigenvalue lambda_minus (column vector)
    """  

    lambda_plus,lambda_minus = lambda_pm(N,gamma,kappa)

    a_plus = (- np.conj(lambda_plus) - gamma*(N -1))/(gamma*np.sqrt(N-1))
    b_plus = 1
   
    lambdaL_plus = np.array([a_plus,b_plus])

    a_minus = (- np.conj(lambda_minus) - gamma*(N -1))/(gamma*np.sqrt(N-1))
    b_minus = 1
    lambdaL_minus = np.array([a_minus,b_minus])

    return lambdaL_plus, lambdaL_minus


def overlap(N,gamma,kappa):

    """
    Compute the analytical expresion of the overlaps with initial uniform superposition state. The overlaps corresponds to each term
    in the analytical expression of survival probability (see text for more details). 

    Args:
        N (int): Number of nodes in complete graph (system size).
        gamma (float): Prefactor scaling the QWer evolution; controls the 'hopping strength'. Optimum for CD at gamma*N = 1.
        kappa (float): Loss rate / strength of incoherent term in the oracle part of the Hamiltonian.

    Returns:
        overlap_plus (float): Overlap correspond to the decay rate governed by Im[lambda_plus].
        overlap_minus (float): Overlap correspond to the decay rate governed by Im[lambda_minus].
        overlap_pm (complex): Overlap correspond to the term having cross terms from both eigenvalues.
    """  

    lambdaR_plus, lambdaR_minus = lambdaR(N,gamma,kappa)
    lambdaL_plus, lambdaL_minus = lambdaL(N,gamma,kappa)

    #initial state
    ket_s = np.array([1.0/np.sqrt(N) ,np.sqrt((N-1)/N)])

    overlap_plus =(1/np.abs(np.vdot(lambdaL_plus,lambdaR_plus))**2)*np.vdot(lambdaR_plus,lambdaR_plus)*np.abs(np.vdot(ket_s, lambdaL_plus))**2

    overlap_minus =(1/np.abs(np.vdot(lambdaL_minus,lambdaR_minus))**2)*np.vdot(lambdaR_minus,lambdaR_minus)*np.abs(np.vdot(ket_s, lambdaL_minus))**2

    overlap_pm = (1/(np.vdot(lambdaR_plus,lambdaL_plus)*np.vdot(lambdaL_minus,lambdaR_minus)))*np.vdot(ket_s,lambdaL_plus)*np.vdot(lambdaL_minus,ket_s)*np.vdot(lambdaR_plus, lambdaR_minus)

    return overlap_plus, overlap_minus,overlap_pm

def surv_prob_theory_total(Tcutoff, dt, N, gamma, kappa):

    """
    Compute the survival probability as a function of time analytically in two-dimensional basis without any reset.
    
    Args:
        Tcutoff (float): Cutoff time for the dynamics.
        dt (float): Discretization of time into time steps of duration dt.
        N (int): Number of nodes in complete graph (system size).
        gamma (float): Prefactor scaling the QWer evolution; controls the 'hopping strength'. Optimum for CD at gamma*N = 1.
        kappa (float): Loss rate / strength of incoherent term in the oracle part of the Hamiltonian.
        
    Returns:
        np.ndarray: Survival probabilities.
    """
 
    m = int(Tcutoff / dt)  # Number of time steps, each with duration dt

    #eigenvalues
    lambda_plus,lambda_minus = lambda_pm(N,gamma,kappa)

    #overlaps 
    overlap_plus, overlap_minus,overlap_pm = overlap(N,gamma,kappa)
   
    s_prob = np.zeros(m) # Initialize array

    for j in range(m): 
        first_term  = np.exp(-1.0j*(lambda_plus - np.conj(lambda_plus))*j*dt)*overlap_plus
        second_term = np.exp(-1.0j*(lambda_minus - np.conj(lambda_minus))*j*dt)*overlap_minus
        third_term  = 2*np.real(np.exp(-1.0j*(lambda_minus - np.conj(lambda_plus))*j*dt)*overlap_pm)

        s_prob[j] = first_term + second_term +third_term


    return s_prob


def surv_prob_theory_total_logspace(Tstep, dt, N, gamma, kappa):

    """
    Compute the survival probability as a function of time analytically in two-dimensional basis without any reset.
    
    Args:
        Tcutoff (float): Cutoff time for the dynamics.
        dt (float): Discretization of time into time steps of duration dt.
        N (int): Number of nodes in complete graph (system size).
        gamma (float): Prefactor scaling the QWer evolution; controls the 'hopping strength'. Optimum for CD at gamma*N = 1.
        kappa (float): Loss rate / strength of incoherent term in the oracle part of the Hamiltonian.
        
    Returns:
        np.ndarray: Survival probabilities.
    """
 
       #eigenvalues
    lambda_plus,lambda_minus = lambda_pm(N,gamma,kappa)

    #overlaps 
    overlap_plus, overlap_minus,overlap_pm = overlap(N,gamma,kappa)
   
    s_prob = np.zeros(len(Tstep)) # Initialize array

    for j,jval in enumerate(Tstep): 
        first_term  = np.exp(-1.0j*(lambda_plus - np.conj(lambda_plus))*jval*dt)*overlap_plus
        second_term = np.exp(-1.0j*(lambda_minus - np.conj(lambda_minus))*jval*dt)*overlap_minus
        third_term  = 2*np.real(np.exp(-1.0j*(lambda_minus - np.conj(lambda_plus))*jval*dt)*overlap_pm)

        s_prob[j] = first_term + second_term +third_term


    return s_prob

def surv_prob_theory_T(T, N, gamma, kappa):

    """
    Compute the survival probability at a time T analytically in two-dimensional basis without any reset.
    
    Args:
        Tcutoff (float): Cutoff time for the dynamics.
        dt (float): Discretization of time into time steps of duration dt.
        N (int): Number of nodes in complete graph (system size).
        gamma (float): Prefactor scaling the QWer evolution; controls the 'hopping strength'. Optimum for CD at gamma*N = 1.
        kappa (float): Loss rate / strength of incoherent term in the oracle part of the Hamiltonian.
        
    Returns:
       sprob: Survival probability at time T.
    """
 
    #eigenvalues
    lambda_plus,lambda_minus = lambda_pm(N,gamma,kappa)

    #overlaps 
    overlap_plus, overlap_minus,overlap_pm = overlap(N,gamma,kappa)
   

    first_term  = np.exp(-1.0j*(lambda_plus - np.conj(lambda_plus))*T)*overlap_plus
    second_term = np.exp(-1.0j*(lambda_minus - np.conj(lambda_minus))*T)*overlap_minus
    third_term  = 2*np.real(np.exp(-1.0j*(lambda_minus - np.conj(lambda_plus))*T)*overlap_pm)

    s_prob = first_term + second_term +third_term


    return s_prob



def find_transition_point(arr,tstep, dt,tp): 
    """
    Find the first time index where the survival probability (SP) drops below tp.
    
    Args:
        arr (numpy array): Array of survival probability values.
        dt (float): Discretization of time into time steps of duration dt.
        
    Returns:
        float: Time at which SP drops below the value of tp.
    """
    idx = np.where(arr <= tp)[0]
    return tstep[idx[0]] * dt if len(idx) > 0 else 0  # Return first occurrence or 0


def overlap_reset(N,gamma,kappa,ket_s):

    """
    Compute the analytical expresion of the overlaps with initial uniform superposition state. The overlaps corresponds to each term
    in the analytical expression of survival probability (see text for more details). 

    Args:
        N (int): Number of nodes in complete graph (system size).
        gamma (float): Prefactor scaling the QWer evolution; controls the 'hopping strength'. Optimum for CD at gamma*N = 1.
        kappa (float): Loss rate / strength of incoherent term in the oracle part of the Hamiltonian.
        ket_s (np.ndarray): the reset state

    Returns:
        overlap_plus (float): Overlap correspond to the decay rate governed by Im[lambda_plus].
        overlap_minus (float): Overlap correspond to the decay rate governed by Im[lambda_minus].
        overlap_pm (complex): Overlap correspond to the term having cross terms from both eigenvalues.
    """  

    lambdaR_plus, lambdaR_minus = lambdaR(N,gamma,kappa)
    lambdaL_plus, lambdaL_minus = lambdaL(N,gamma,kappa)

    overlap_plus =(1/np.abs(np.vdot(lambdaL_plus,lambdaR_plus))**2)*np.vdot(lambdaR_plus,lambdaR_plus)*np.abs(np.vdot(ket_s, lambdaL_plus))**2

    overlap_minus =(1/np.abs(np.vdot(lambdaL_minus,lambdaR_minus))**2)*np.vdot(lambdaR_minus,lambdaR_minus)*np.abs(np.vdot(ket_s, lambdaL_minus))**2

    overlap_pm = (1/(np.vdot(lambdaR_plus,lambdaL_plus)*np.vdot(lambdaL_minus,lambdaR_minus)))*np.vdot(ket_s,lambdaL_plus)*np.vdot(lambdaL_minus,ket_s)*np.vdot(lambdaR_plus, lambdaR_minus)

    return overlap_plus, overlap_minus,overlap_pm


def surv_prob_det_reset(Tcutoff, dt, r, N, gamma, kappa):
    
    """
    Compute the survival probability as a function of time using analytics in two dimensional basis. Here reset is done using
    deterministic protocol to the initial state.
    
    Args:
        Tcutoff (float): Cutoff time for the dynamics.
        dt (float): Discretization of time into time steps of duration dt.
        r (int): reset value (such that the time after which we reset is t_reset = r*dt).
        N (int): Number of nodes in complete graph (system size).
        gamma (float): Prefactor scaling the QWer evolution; controls the 'hopping strength'. Optimum for CD at gamma*N = 1.
        w (int): Target node/site location; dynamics invariant under choice of w.
        kappa (float): Loss rate / strength of incoherent term in the oracle part of the Hamiltonian.
        
    Returns:
        np.ndarray: Survival probabilities.
    """
  
    m = int(Tcutoff / dt)
    s_prob_reset = np.zeros(m)

    # Initial state
    initial_state =np.array([1.0/np.sqrt(N), np.sqrt((N-1)/N)])
    reset_state = initial_state.copy()

    # Set initial survival probability
    s_prob_reset[0] = 1

    # Precompute eigenvalues and overlaps
    lambda_plus, lambda_minus = lambda_pm(N, gamma, kappa)
    overlap_plus, overlap_minus, overlap_pm = overlap_reset(N, gamma, kappa, reset_state)

    t_r = 0

    for j in range(1, m):

        first_term  = np.exp(-1.0j*(lambda_plus - np.conj(lambda_plus))*(j*dt -t_r))*overlap_plus
        second_term = np.exp(-1.0j*(lambda_minus - np.conj(lambda_minus))*(j*dt -t_r))*overlap_minus
        third_term  =  2*np.real(np.exp(-1.0j*(lambda_minus - np.conj(lambda_plus))*(j*dt -t_r))*overlap_pm)

        s_prob_reset[j] = np.real(first_term + second_term + third_term)

        #reset times 
        if r != 0 and j % r == 0:
            t_r =  j*dt
            reset_state = np.sqrt(s_prob_reset[j])* np.array([1.0/np.sqrt(N), np.sqrt((N-1)/N)])
            overlap_plus, overlap_minus, overlap_pm = overlap_reset(N, gamma, kappa, reset_state)
    

    return s_prob_reset




def alpha_of_rs_scalar(rbar, s):
    
    """
    Compute the exponent alpha of the time complexity of the search process.
    
    Args:
        rbar (float): Hopping/Tunnelling exponent i.e. gamma = \bar{gamma} N^{-rbar -1} (see main text)
        s (float): Monitoring Exponent i.e. kappa = \bar{kappa} N^{-s} (see main text)
       
    Returns:
        alpha(float): Time Complexity Exponent i.e. tau = \Theta(N^{alpha})
    """

    if abs(rbar - 0.0) < 1e-3:
        return s if s >= 0.5 else 1.0 - s
        ""
    elif rbar > 0.0:
        if s >= 0:
            return 2.0*rbar + s + 1.0

        else:
            return 2.0*rbar - s + 1.0

    else: # rbar < 1
        if s >= rbar:
            return 1.0 + s

        elif s < rbar:
            return 2.0*rbar - s + 1.0


def Heff_matrix(gamma, kappa, N):
    """
    Compute the 2x2 effective system Hamiltonian matrix H_eff^(s) in the reduced
    two-dimensional basis {|w>, |r_perp>}.

    Details:
    --------
    This function constructs the matrix
    H = [[a, b],
    [b, d]]
    with
    a = - (gamma + 1 + i * kappa),
    b = - gamma * sqrt(N - 1),
    d = - gamma * (N - 1).

    These definitions follow the effective Hamiltonian used to describe the
    search dynamics on a complete graph, with the target-site energy set to 1
    and time measured in the model's dimensionless units.

    Parameters:
    -----------
    gamma : (float) Tunneling (hopping) strength, gamma > 0.
    kappa : (float) Monitoring (decay/detection) rate, kappa >= 0.
    N : (int) Number of nodes in the complete graph (Hilbert-space dimension). Expect integer N >= 1.

    Returns:
    --------
    H : numpy.ndarray, shape (2, 2), dtype complex
    The complex 2x2 matrix representing the effective Hamiltonian in the
    reduced basis. H[0,0] = a, H[0,1] = H[1,0] = b, H[1,1] = d.

    Notes:
    ------
    The matrix is in general non-Hermitian because the diagonal element, a,
    contains the imaginary term -i * kappa. This non-Hermiticity models
    irreversible detection/monitoring effects.

    For N = 1 the off-diagonal b is zero. The function avoids taking the
    square root of a negative number when called with N < 1, but callers
    should supply sensible N (typically N >= 2 for the physical model).
    """
    a = -(gamma + 1.0 + 1j*kappa)
    b = -gamma*np.sqrt(max(N-1,0.0))
    d = -gamma*(N-1)
    return np.array([[a, b],[b, d]], dtype=complex)

def eig_biorth(H):
    """
    Compute the biorthogonal eigendecomposition for a (possibly non-Hermitian, but symmetric)
    2x2 matrix H.

    Details:
    --------
    This function performs the following:
    1. Computes eigenvalues and right eigenvectors of H using a standard numerical
    eigendecomposition of numpy.
    2. Sorts the eigenpairs by descending imaginary part of the (complex) eigenvalue,
    such that lam[0] is the "slow mode" and lam[1] is the "fast mode"; see article supplemental text.
    3. Returns the right eigenvector matrix with eigenvectors as columns, and a
    left-eigenvector representation obtained by transposing the right-eigenvector
    matrix (VR.T). The transpose choice follows the symmetry assumption; left eigenvectors
    must be modified when treating a generic NH matrix. 

    Parameters:
    -----------
    H : (numpy.ndarray, shape (2, 2)) Symmetric square matrix, may be complex and non-Hermitian.

    Returns:
    --------
    lam : numpy.ndarray, shape (2,), dtype complex
    Eigenvalues of H, sorted by decreasing imaginary part.
    VR : numpy.ndarray, shape (2, 2), dtype complex
    Right eigenvectors stored as columns: VR[:, i] corresponds to lam[i].
    VL : numpy.ndarray, shape (2, 2), dtype complex
    Left eigenvectors stored as rows: VL[i,:] corresponds to lam[i].

    Notes:
    ------
    NumPy's eigenvector phases and normalizations are arbitrary; we do not enforce a specific
    normalization, since vectors being normalized is sufficient for our purposes. One can 
    verify that condition <L_i|R_j> = delta_ij is fulfilled.

    Care should be taken if parameters of matrix H are chosen such that we are at or near 
    an exceptional point where eigenvectors coalesce.

    For a generic non-symmetric matrix, the true left eigenvectors are not the
    simple transpose of the right eigenvectors; this function follows the
    convention used in our article where the matrix H exhibits the
    required symmetry.
    """
    # Compute eigenvalues and right eigenvectors
    lam, VR = np.linalg.eig(H)
    # Sorting (used in other functions; see description above)
    idx = np.argsort(-np.imag(lam))
    lam = lam[idx]; VR = VR[:, idx]
    # Obtain left eigenvectors from the right eigenvectors
    VL = VR.T
    return lam, VR, VL

def overlaps_from_bi(VR, VL, N):
    """
    Compute overlap weights (i.e. prefactors of the exponential terms) that enter the 
    no-click probability using a biorthogonal eigenbasis.

    Details:
    --------
    This function computes three quantities derived from the left and right
    eigenvectors:
    O_f : overlap weight associated with the "fast" eigenmode (eigenvalue with the largest absolute imaginary part).
    O_s : overlap weight associated with the "slow" eigenmode (eigenvalue with the smallest absolute imaginary part).
    O_pm: overlap for the interference term (comes from off-diagonal terms in no-click probability).

    Procedure:
    1. Construct the two-component representation of the uniform initial state s:
    s = [1/sqrt(N), sqrt(1 - 1/N)].
    2. Extract right eigenvectors R0, R1 (columns of VR) and left eigenvectors L0, L1 (rows of VL).
    3. Compute squared norms of right eigenvectors and inner products of left
    eigenvectors with s to build the diagonal overlap weights.
    4. Compute the off-diagonal combination that multiplies the oscillatory
    interference term in the no-click probability expansion.

    Parameters:
    -----------
    VR : (numpy.ndarray, shape (2, 2)) Right eigenvectors |i> as columns (VR[:, i] is the right eigenvector for lam[i]).
    VL : (numpy.ndarray, shape (2, 2)) Left eigenvectors <i| as rows (VL[i,:] used as the left eigenvector for lam[i]).
    N : (int) Hilbert-space dimension used to form the uniform state s. Expected N >= 1.

    Returns:
    --------
    O_f : (float)
    Overlap weight for the fast eigenmode (non-negative and real).
    O_s : (float)
    Overlap weight for the slow eigenmode (non-negative and real).
    O_pm : (float)
    Interference term overlap

    Notes:
    ------
    The function uses the standard Hermitian inner product for vector norms and
    overlaps. Small numerical imaginary parts due to floating point rounding may
    appear; they are not removed by this function.

    The mapping from indices {0,1} to labels {slow, fast} assumes the
    ordering convention used when sorting eigenvalues in function eig_biorth.
    """
    # Initial uniform superposition state
    s = np.array([1/np.sqrt(N), np.sqrt(1-1/N)], dtype=complex)
    # Making right- and left-eigenvectors explicit
    R0 = VR[:,0]; R1 = VR[:,1]
    L0 = VL[0,:]; L1 = VL[1,:]
    # Function for computing norm of vector
    def norm_sq(v): return float(np.real_if_close(np.vdot(v,v)))
    # Compute overlaps as per analytic expressions (see article for further details)
    O_s = abs(np.vdot(L0, s))**2 * norm_sq(R0)
    O_f = abs(np.vdot(L1, s))**2 * norm_sq(R1)
    O_pm = (np.vdot(L1, s)) * (np.vdot(L0, s).conjugate()) * (np.vdot(R1, R0))
    return O_f, O_s, O_pm

def overlaps_analytic(gbar, r_exp, kbar, s_exp, N):
    """
    Compute exact and approximate analytic overlap expressions for the two effective
    eigenmodes for given values of hopping and monitoring strengths, gamma and kappa.

    Details:
    --------
    This definition trivially implements closed-form expressions and simple asymptotic
    approximations (see detailed calculations in our article) for the overlaps associated
    with the "fast" and "slow" modes, given the parameter scalings:
    gamma = gbar * N^(-r_exp),
    kappa = kbar * N^(-s_exp).

    Procedure outlined:
    1. Compute gamma and kappa from the supplied prefactors and exponents.
    2. Form the 2x2 effective Hamiltonian matrix elements a, b, d.
    3. Compute exact eigenvalues lambda_plus and lambda_minus from the quadratic
    formula.
    4. Compute the parameters entering the eigenvectors v_plus and v_minus defined as (lambda - d)/b.
    5. Use heuristic/expanded expressions v_f and v_s for the fast and slow modes.
    6. Evaluate overlap function (analytic; supplied in article supplemental material) using
    v values (exact/approximate) to obtain overlaps.

    Parameters:
    -----------
    gbar : (float) Prefactor in hopping strength parameter gamma.
    r_exp : (float) Exponent in gamma = gbar * N^{-r_exp}.
    kbar : (float) Prefactor in monitoring strength parameter kappa.
    s_exp : (float) Exponent in kappa = kbar * N^{-s_exp}.
    N : (int) Hilbert-space dimension (complete graph size). Use N >= 2.

    Returns:
    --------
    O_f : (float)
    Analytic approximation (asymptotic, for large N) for the overlap associated with the fast mode.
    O_s : (float)
    Analytic approximation (asymptotic, for large N) for the overlap associated with the slow mode.
    O_plus : (float)
    Overlap computed from the exact eigenvector component v_plus.
    O_minus : (float)
    Overlap computed from the exact eigenvector component v_minus.

    Notes:
    ------
    Returned values are cast to real numbers because the overlap expressions are
    expected to be real in the physically relevant parameter regimes; small
    imaginary parts due to numerical rounding are discarded.

    We assume b != 0 and 1 + v^2 != 0; if these denominators approach zero results
    may be unreliable and such cases should be handled explicitly.

    This routine is intended solely to provide analytic confirmation, verifying our derivation 
    and checking asymptotic behavior. For precise numerical evaluation use the numerical pipeline:
    Heff_matrix -> eig_biorth -> overlaps_from_bi.
    """
    # Parameters for hopping and monitoring, including N-scaling
    gamma = gbar * N**(-r_exp)
    kappa = kbar * N**(-s_exp)

    # Matrix elements of the effective Hamiltonian
    a = -(gamma + 1 + 1j * kappa)
    b = -gamma * np.sqrt(N - 1)
    d = -gamma * (N - 1)

    # Eigenvalues λ± of effective 2x2 system Hamiltonian (exact)
    delta = (a - d) / 2
    sqrt_term = np.sqrt(delta**2 + b**2)
    lam_plus  = (a + d) / 2 + sqrt_term
    lam_minus = (a + d) / 2 - sqrt_term

    # v± = (λ± − d)/b (eigenvector (unnormalized) components; exact)
    v_plus  = (lam_plus  - d) / b
    v_minus = (lam_minus - d) / b

    # Fast and slow eigenvector components (based on asymptotic expansion of λ± for N->infinity)
    v_f  = (a  - d) / b + b / (a - d)
    v_s = -b / (a - d)

    # Analytic expression for the overlaps (exact) (excludes the interference term!)
    def get_overlap(v):
        num = (1 + np.abs(v)**2) * np.abs((np.sqrt(1-1/N))+v/np.sqrt(N))**2
        den = np.abs(1+v**2)**2
        return num / den
    # Compute O_plus and O_minus - these are analytically exact.
    O_plus  = get_overlap(v_plus)
    O_minus = get_overlap(v_minus)
    # Compute O_f and O_s from the asypmtotic approximations for the vector components. (Accurate for large N.)
    O_f  = get_overlap(v_f)
    O_s = get_overlap(v_s)

    return O_f.real, O_s.real, O_plus.real, O_minus.real  # Should be real-valued



