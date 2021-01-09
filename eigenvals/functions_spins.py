"""
Copyright (c) 2018 and later, Muhammad Irfan and Anton Akhmerov.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
"""

from functools import partial

import kwant
from kwant.digest import uniform
import numpy as np
import tinyarray
import scipy.linalg as la
from scipy import integrate
import scipy.sparse.linalg as sla


def make_system(a, W, L, Wsc, par):
    """Make a normal scattering region (2DEG) 
    attached to two translationally invariant SC leads.

    Parameters
    ----------
    a : integer
        Lattice constant of a square lattice.
    W : integer
        Width of the leads (along the y-axis) attached to the scattering region.
    L : integer
        Length of the scattering region along the x-axis between the leads.
    Wsc : integer
        Width of the SC region along the y-axis.
    """
    def N_shape(pos):
        (x, y) = pos
        return -W / 2 <= y <= W / 2 and -L / 2 <= x <= L / 2

#     def SC_shape(pos):
#         (x, y) = pos
#         return (W / 2 < y <= (W/2 + Wsc) or -(W/2 + Wsc) <= y < -W/ 2) and -L / 2 <= x <= L / 2

    def SC_shape_upper(pos):
        (x, y) = pos
        return (W / 2 < y <= (W/2 + Wsc)) and -L / 2 <= x <= L / 2

    def SC_shape_lower(pos):
        (x, y) = pos
        return (-(W/2 + Wsc) <= y < -W/ 2) and -L / 2 <= x <= L / 2

    s0 = np.eye(2)
    sz = np.array([[1, 0], [0, -1]])
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    
    szs0 = np.kron(sz, s0)
    sxs0 = np.kron(sx, s0)
    sys0 = np.kron(sy, s0)
    s0sz = np.kron(s0, sz)
    s0sx = np.kron(s0, sx)
    sysz = np.kron(sy, sz)
    sxsz = np.kron(sx, sz)
    
    parameters = par
    
    
    def onsite(site, par):   
        bar = 0
        return ( 4 * par.t - par.mu + bar ) * s0sz + par.Bz * 0.5 * 0.5 * par.g * szs0

    def onsite_sc_lower(site, par):
        delta_s0sx = np.kron(s0, np.array([[0, par.delta * np.exp(-1j * par.phi/2.0)], [par.delta * np.exp(1j * par.phi/2.0), 0]]))
        return ( 4 * par.t - par.mu ) * s0sz + delta_s0sx

    def onsite_sc_upper(site, par):
        delta_s0sx = np.kron(s0, np.array([[0, par.delta * np.exp(1j * par.phi/2.0)], [par.delta * np.exp(-1j * par.phi/2.0), 0]]))
        return ( 4 * par.t - par.mu ) * s0sz + delta_s0sx #+ par.e_z * sxs0

    def hop_sc(site1, site2, par):
        return -par.t * s0sz
    
    def hopx(site1, site2, par=None):
        if(par is None):
            par = parameters
        hop = 0
        if (N_shape(site1.pos) and N_shape(site2.pos)):
            hop = (-par.t * s0sz  - 1j * par.tSO * sysz) #* np.exp(1j * (y1) * (x1 - x2) * par.Bz ) # Peierls? Tylko znak Bz dla dziur inny...
        else:
            hop = -par.t * s0sz
        return hop
       

    def hopy(site1, site2, par=None):
        if(par is None):
            par = parameters
            
        return -par.t * s0sz  + 1j * par.tSO * sxsz


    lat = kwant.lattice.square(a, norbs=4)
    syst = kwant.Builder()
    syst[lat.shape(N_shape, (0, 0))] = onsite
#     syst[lat.shape(SC_shape, (0, W/2 + Wsc/2 ))] = onsite_sc_upper
#     syst[lat.shape(SC_shape, (0,-W/2 - Wsc/2))] = onsite_sc_lower
    syst[lat.shape(SC_shape_upper, (0, W/2 + Wsc/2 ))] = onsite_sc_upper
    syst[lat.shape(SC_shape_lower, (0,-W/2 - Wsc/2))] = onsite_sc_lower
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy
    # tutaj z tego tutotialu: https://kwant-project.org/doc/1/tutorial/first_steps

    def lead_sc_shape(pos):
        (x, y) = pos
        return -L / 2 <= x <= L / 2

    
    syst = syst.finalized()
    return syst


def sparse_eigs(ham, n_eigs, n_vec_lanczos, sigma=0):
    """Compute eigenenergies using MUMPS as a sparse solver.

    Parameters:
    ----------
    ham : coo_matrix
        The Hamiltonian of the system in sparse representation..
    n_eigs : int
        The number of energy eigenvalues to be returned.
    n_vec_lanczos : int
        Number of Lanczos vectors used by the sparse solver.
    sigma : float
        Parameter used by the shift-inverted method. See
        documentation of scipy.sparse.linalg.eig

    Returns:
    --------
    A list containing the sorted energy levels. Only positive
    energies are returned.
    """
    class LuInv(sla.LinearOperator):
        def __init__(self, A):
            inst = kwant.linalg.mumps.MUMPSContext()
            inst.factor(A, ordering='metis')
            self.solve = inst.solve
            try:
                super(LuInv, self).__init__(shape=A.shape, dtype=A.dtype,
                                            matvec=self._matvec)
            except TypeError:
                super(LuInv, self).__init__(shape=A.shape, dtype=A.dtype)

        def _matvec(self, x):
            return self.solve(x.astype(self.dtype))

    ev, evecs = sla.eigs(ham, k=n_eigs,
                         OPinv=LuInv(ham), sigma=sigma, ncv=n_vec_lanczos)

    energies = list(ev.real)
    return energies, evecs
