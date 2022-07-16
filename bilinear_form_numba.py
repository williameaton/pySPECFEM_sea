from numba import int32, int64, boolean, float64
from numba import njit, jit, typeof
import numpy as np

@njit(float64(float64, float64))
def d(i, j):
    # Kronecker delta using Numba format
    if i == j:
        return 1.0
    else:
        return 0.0



@njit(float64[:,:,:](float64[:,:], float64[:], int32, int32, float64, float64 ))
def _calc_BF_fluid_internal_numba(detjac, gll_weights, ngll, nelem, g_1, del_n_rho):
    # Calculates the matrix for the fluid gravitational perturbation using Numba
    F = np.zeros((nelem, ngll, ngll))
    for i_elem in range(nelem):
        for abg in range(ngll):
            for stn in range(ngll):
                for bars in range(ngll):
                    F[i_elem, abg, stn] += g_1 * del_n_rho * d(bars, abg) *  d(bars, stn) * detjac[bars, i_elem] * gll_weights[bars]
    return F





@njit(float64( int64, int64, int64, int64, int64, int64, float64[:,:,:], float64[:,:,:,:]))
def _get_D_internal_sum_numba(i, j, stn, abg, bars, i_elem, dlagrange_gll, jacinv):
    internalsum = 0
    for k in range(3):
        for r in range(3):
            for q in range(3):
                for p in range(3):
                    t1 = ((d(r,i) * jacinv[k, q, bars, i_elem]) + (d(q,i) * jacinv[k, r, bars, i_elem]) - ( (2/3) * d(r,q) * d(i,k) * jacinv[k, i, bars, i_elem])) * dlagrange_gll[k, bars, abg]
                    t2 = ((d(r,j) * jacinv[p, q, bars, i_elem]) + (d(q,j) * jacinv[p, r, bars, i_elem]) - ( (2/3) * d(r,q) * d(j,p) * jacinv[p, j, bars, i_elem])) * dlagrange_gll[p, bars, stn]

                    internalsum += t1*t2
    return internalsum



@njit(float64[:,:,:](float64[:,:], float64[:], int32, int32, float64[:,:], float64[:,:,:,:], float64[:,:,:] ))
def _calc_BF_strain_deviator_numba(detjac, gll_weights, ngll, nelem,  shearmod, jacinv, dlagrange_gll):
    D = np.zeros((nelem, ngll*3, ngll*3))

    for i_elem in range(nelem):
        m = 0
        for abg in range(ngll):
            for i in range(3):
                n = 0
                for stn in range(ngll):
                    for j in range(3):
                        bars_sum = 0
                        for bars in range(ngll):
                            pi = detjac[bars, i_elem] * gll_weights[bars]
                            mu = shearmod[bars, i_elem]
                            sum = _get_D_internal_sum_numba(i, j, stn, abg, bars, i_elem, dlagrange_gll[:,:,:], jacinv[:,:,:,:])

                            bars_sum += (sum*pi*mu)
                        D[i_elem, m, n] = bars_sum
                        n += 1
                m += 1
    return D*0.5000000000000

