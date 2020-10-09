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


def make_system(a, W, L, Wsc):
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

    def SC_shape(pos):
        (x, y) = pos
        return (W / 2 < y <= (W/2 + Wsc) or -(W/2 + Wsc) <= y < -W/ 2) and -L / 2 <= x <= L / 2

#     sigma_0 = tinyarray.array([[1, 0], [0, 1]])
#     sigma_x = tinyarray.array([[0, 1], [1, 0]])
#     sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
#     sigma_z = tinyarray.array([[1, 0], [0, -1]])

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
    
    
    def onsite(site, par):
        bar = 0
#         if abs(x - W/2)<4:
#             bar=0.3
        return ( 4 * par.t - par.mu + bar ) * s0sz + par.e_z * sxs0
#         return ( 4 * par.t - par.mu + bar ) * sz

    def onsite_sc_lower(site, par):
#         return ( 4 * par.t - par.mu_sc ) * s0sz + par.e_z * sxs0 + par.delta * s0sx
        delta_s0sx = np.kron(s0, np.array([[0, par.delta * np.exp(-1j * par.phi/2.0)], [par.delta * np.exp(1j * par.phi/2.0), 0]]))
        return ( 4 * par.t - par.mu ) * s0sz + par.e_z * sxs0 + delta_s0sx
#         return ( 4 * par.t - par.mu ) * sz + par.delta * sx

    def onsite_sc_upper(site, par):
#         return ( 4 * par.t - par.mu_sc ) * s0sz + par.e_z * sxs0 + par.delta * s0sx
        delta_s0sx = np.kron(s0, np.array([[0, par.delta * np.exp(1j * par.phi/2.0)], [par.delta * np.exp(-1j * par.phi/2.0), 0]]))
        return ( 4 * par.t - par.mu ) * s0sz + par.e_z * sxs0 + delta_s0sx
#         return ( 4 * par.t - par.mu ) * sz + par.delta * sx

    
    def hopx(site1, site2, par):
        return -par.t * s0sz  - 1j * par.alpha * sysz
#         return -par.t * sz 

    def hopy(site1, site2, par):
        return -par.t * s0sz  + 1j * par.alpha * sxsz
#         return -par.t * sz 


    lat = kwant.lattice.square(a)
#     lat = kwant.lattice.square(a, norbs=)
    syst = kwant.Builder()
    syst[lat.shape(N_shape, (0, 0))] = onsite
    syst[lat.shape(SC_shape, (0, W))] = onsite_sc_upper
    syst[lat.shape(SC_shape, (0, -W))] = onsite_sc_lower
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy
#     lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)), conservation_law=-sz)
#     lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)), time_reversal= 1j * sigma_y)

#     def lead_shape(pos):
#         (x, y) = pos
#         return -W / 4 <= y <= W / 4

#     def lead_hopx(site1, site2, par):
#         return -par.t * s0sz  - 1j * par.alpha * sysz
# #         return -par.t * sz

#     def lead_onsite(site, par):
# #         (x, y) = site.pos
# #         tip = par.U0 / ( (( y - par.ytip)**2 + (x-par.xtip)**2)/par.dtip**2 + 1 )
#         return ( 4 * par.t - par.mu ) * s0sz + par.e_z * sxs0
# #         return ( 4 * par.t - par.mu ) * sz #+ par.e_z * sigma_x

#     lead[lat.shape(lead_shape, (-1, 0))] = lead_onsite
#     lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = lead_hopx
#     lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy
#     syst.attach_lead(lead)
#     syst.attach_lead(lead.reversed())


    def lead_sc_shape(pos):
        (x, y) = pos
        return -L / 2 <= x <= L / 2

#     def lead_hopx(site1, site2, par):
#         return -par.t * s0sz  - 1j * par.alpha * sysz

#     lead_sc = kwant.Builder(kwant.TranslationalSymmetry((0, -a)))
#     lead_sc[lat.shape(lead_sc_shape, (0, -1))] = onsite_sc_lower
#     lead_sc[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx
#     lead_sc[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy
#     syst.attach_lead(lead_sc)
    
#     lead_sc = kwant.Builder(kwant.TranslationalSymmetry((0, a)))
#     lead_sc[lat.shape(lead_sc_shape, (0, 1))] = onsite_sc_upper
#     lead_sc[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx
#     lead_sc[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy
#     syst.attach_lead(lead_sc)

     # Here, we make the scattering states of the leads purely electron or holes.
    # This is necessary to calculate the Andreev conductance.
    # Projectors - P_1 projects out the electron part, and P_2
    # the hole part of the Hamiltonian.
#     P_1 = np.array([[1, 0], [0, 0], [0, 1], [0, 0]])
#     P_2 = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
#     projector_list = [P_1, P_2]
#     syst.leads[0] = cons_leads.ConservationInfiniteSystem(
#         syst.leads[0], projector_list)
    
    syst = syst.finalized()
    return syst


# def calculate_energies(par_closed):
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
    return energies

    
#     sys = make_system(par_closed, plot = False)
#     ham_mat = sys.hamiltonian_submatrix(args=[par_closed], sparse=True)
    

#     #ev = sla.eigsh(ham_mat, k=20, which='LM', sigma=0.0, return_eigenvectors=False)
    
#     n_values = 30
#     ev = sparse_eigs(ham_mat, n_eigs=n_values, n_vec_lanczos=3*n_values+10, sigma=0.0)

#     return np.asarray(ev)