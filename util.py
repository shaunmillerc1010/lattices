import numpy as np
import math
import fpylll
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

def create_lwe_instance(n = 8, sigma = 1, prime_modulus = 521, samples = 8):
    a_0 = np.random.uniform(low = 0, high = prime_modulus - 1, size = n).round()
    s = np.random.normal(loc = 0, scale = sigma, size = n).round()
    e_0 = round(np.random.normal(loc = 0, scale = sigma))
    b_0 = (a_0.dot(s) + e_0) % prime_modulus
    A, b, e, s = [list(a_0)], [b_0], [e_0], list(s)
    for _ in range(1,samples):
        a_i = np.random.uniform(low = 0, high = prime_modulus - 1, size = n).round()
        e_i = round(np.random.normal(loc = 0, scale = sigma))
        A, e, b = A + [list(a_i)], e + [int(e_i)], b + [(a_i.dot(s) + e_i) % prime_modulus]
    e = [-1 * entry for entry in e]
    v = [int(v_i) for v_i in s + e + [-1]]
    return A, list(b), list(v)

def lwe_to_primal(A, b, prime_modulus):
    A, b = np.array(A), np.array(b)
    nrow, mcol = np.shape(A.transpose())
    qI_m = prime_modulus * np.identity(mcol)
    I_n, zero_m_n = np.identity(nrow), np.zeros((mcol, nrow))
    pad = np.array([[0 for i in range(nrow + mcol)] + [1]])
    zeros_n = np.array([0 for i in range(nrow)])
    C = np.block([[I_n, A.transpose()],[zero_m_n,qI_m],[zeros_n,b]])
    C = np.block([C, pad.transpose()])
    C = C.astype(int)
    return C.tolist()

def list_to_file(some_list, some_path = ''):
    M = fpylll.IntegerMatrix.from_matrix(some_list)
    f = open(some_path, 'w')
    m = str(M)
    f.write(m)
    f.close()
    return 0

def update_until_v_found(M, min_block_size, max_block_size, max_tours, v, prime_modulus):
    L = fpylll.IntegerMatrix.from_matrix(M)
    G = fpylll.GSO.Mat(L)
    G.update_gso()
    Y = BKZ2(G)
    block_size = min_block_size
    v = list(np.array(v) % prime_modulus)
    trial_v = list(np.array(M[0]) % prime_modulus)
    print('completed block sizes: ', sep=' ', end=' ', flush=True)
    while block_size <= max_block_size:
        o = fpylll.BKZ.Param(block_size,strategies = 'default.json', min_success_probability=0.5, flags = fpylll.BKZ.AUTO_ABORT)
        #o = fpylll.BKZ.Param(block_size, min_success_probability=0.5, flags = fpylll.BKZ.AUTO_ABORT) # no preprocessing
        tour = 1
        while (tour < max_tours) & (Y.tour(o) == False):
            tour += 1
            Hold_M = [[0 for i in M] for j in M[0]]
            L.to_matrix(Hold_M)
            trial_v = list(np.array(Hold_M[0]) % prime_modulus)
            neg_trial_v = list(-1 * np.array(Hold_M[0]) % prime_modulus)
            if (trial_v != v) & (neg_trial_v != v):
                L.to_matrix(M)
            else:
                print('....v found at (tour,block size) = ' + str((tour,block_size)))
                return M, block_size
        print(str(block_size), sep=' ', end=' ', flush=True)
        block_size += 1
    print('exhausted... secret not found for block size <= ' + str(max_block_size))
    return M, 0

def approx_2016_primal(q,n,m,beta, sigma, v_norm):
    delta_0 = (((math.pi*beta)**(1/beta)*beta)/(2*math.pi*math.e))**(1/(2*(beta-1)))
    volume = q**((m)/(n+m+1))
    left = (beta/(m+n+1))**(1/2) * v_norm
    right = (delta_0)**(2*beta-(n+m+1)) * (volume)
    return (left) <= right
# when in doubt of estimates, look at https://eprint.iacr.org/2017/815.pdf
def get_gsa_lengths(n, sigma, samples, block_size, prime_modulus):
    beta, m, q = block_size, samples, prime_modulus
    delta_0 = (((math.pi *beta)**(1/beta) * beta)/(2 * math.pi * math.e))**(1/(2*(beta-1)))
    volume = q**((m)/(n+m+1))
    right = (delta_0)**(2*beta-(n+m+1)) * (volume)
    return [delta_0**(n + m + 1 - (2*(i))) * volume for i in range(n+m+1)]

def t_beta_primal(v_list, n, sigma, prime_modulus, samples):
    theoretical_minBeta = []
    for v_norm in v_list:
        beta = 30
        while approx_2016_primal(prime_modulus, n, samples, beta, sigma, v_norm)==False:
            beta +=1
        theoretical_minBeta += [beta]
    return theoretical_minBeta




def get_projection(M,v):
    proj_lengths = [float(np.linalg.norm(v))] #the first projection is just the identity
    print('finding ' + str(len(M)) + ' projections of v...')
    roundoff = 0.05
    #fpylll.FPLLL.set_precision(60)
    for k in range(1,len(M)):
        J = fpylll.IntegerMatrix.from_matrix(M[:k] + [v])
        #G = fpylll.GSO.Mat(J, float_type = 'mpfr')
        G = fpylll.GSO.Mat(J)
        _ = G.update_gso()
        proj = G.get_r(k,k)**(0.5)#get r outputs length^2
        if type(proj) == complex: proj_lengths.append(0)
        else: proj_lengths.append(round(proj,4))
    if 0 in proj_lengths: index_proj_found = proj_lengths.index(0) + 1
    else: index_proj_found = len(M)
    #print('precision used: ' + str(fpylll.FPLLL.get_precision()))
    return proj_lengths, index_proj_found

def get_t_projection(v):
    dim = len(v)
    #outputs theoretical projection lengths given the size of v
    v_length = float(np.linalg.norm(v))
    return [v_length*((dim-i+1)/(dim))**(1/2) for i in range(1, dim+1)]

def get_exp_gso(M, v, max_block_size, prime_modulus):
    M,block_size = update_until_v_found(M, 20, max_block_size, 20, v, prime_modulus)
    J = fpylll.IntegerMatrix.from_matrix(M)
    #G = fpylll.GSO.Mat(J, float_type = 'mpfr')
    G = fpylll.GSO.Mat(J)
    _ = G.update_gso()
    return M, block_size, [G.get_r(k,k)**(0.5) for k in range(len(v))]

def get_GSA_focus( n, sigma, samples, prime_modulus):
    #indexing should begin at done
    block_size = 2
    gsa_focus = [0]
    dim = n+samples+1
    while block_size <= n+samples+1:
        gsa = get_gsa_lengths(n, sigma, samples, block_size, prime_modulus)
        print(gsa)
        gsa_focus += [gsa[dim-block_size]]#note indexing starts at 0 not 1
    return gsa_focus[::-1]
