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


def make_system(a, W, L, dL, Wb):
    def N_shape(pos):
        (x, y) = pos
        return -W / 2 <= y <= W / 2 and -L / 2 <= x <= L / 2

    sigma_0 = tinyarray.array([[1, 0], [0, 1]])
    sigma_x = tinyarray.array([[0, 1], [1, 0]])
    sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
    sigma_z = tinyarray.array([[1, 0], [0, -1]])
    
    def onsite(site, par):
        return  ( 4 * par.t - par.mu ) * sigma_0 + par.e_z * sigma_x

    def hopx(site1, site2, par):
        return -par.t * sigma_0  - 1j * par.alpha * sigma_y

    def hopy(site1, site2, par):
        return -par.t * sigma_0  + 1j * par.alpha * sigma_x


    lat = kwant.lattice.square(a, norbs=2)
    syst = kwant.Builder()
    syst[lat.shape(N_shape, (0, 0))] = onsite
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)), time_reversal= 1j * sigma_y)

    def lead_shape(pos):
        (x, y) = pos
        return -W / 2 <= y <= W / 2

    def lead_hopx(site1, site2, par):
        return -par.t * sigma_0  - 1j * par.alpha * sigma_y

    def lead_onsite(site, par):
        (x, y) = site.pos
        return ( 4 * par.t - par.mu ) * sigma_0 

    lead[lat.shape(lead_shape, (-1, 0))] = lead_onsite
    lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = lead_hopx
    lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    syst = syst.finalized()
    return syst


def supercurrent_tight_binding(smatrix, phi, Delta):
    """Returns the supercurrent in a SNS Josephson junction using
    a tight-binding model.

    Parameters
    ----------
    smatrix : kwant.smatrix object
        Contains scattering matrix and information of lead modes.
    phi : float
        Superconducting phase difference between two superconducting leads.
    Delta : float
        Superconducting gap.
    """
    N, M = [len(li.momenta) // 2 for li in smatrix.lead_info]
    s = smatrix.data
    r_a11 = 1j * np.eye(N)
    r_a12 = np.zeros((N, M))
    r_a21 = r_a12.T
    r_a22 = 1j * np.exp(- 1j * phi) * np.eye(M)
    r_a = np.bmat([[r_a11, r_a12], [r_a21, r_a22]])
    # Matrix
    A = (r_a.dot(s) - (s.T).dot(r_a)) / 2
    # dr_a/dphi
    dr_a11 = np.zeros((N, N))
    dr_a12 = np.zeros((N, M))
    dr_a21 = dr_a12.T
    dr_a22 = np.exp(-1j * phi) * np.eye(M)
    dr_a = np.bmat([[dr_a11, dr_a12], [dr_a21, dr_a22]])

    # dA/dphi
    dA = (dr_a.dot(s) - (s.T).dot(dr_a)) / 2
    # d(A^dagger*A)/dphi
    Derivative = (dA.T.conj()).dot(A) + (A.T.conj()).dot(dA)
    Derivative = np.array(Derivative)
    eigVl, eigVc = la.eigh(A.T.conj().dot(A))
    eigVl = Delta * eigVl ** 0.5
    eigVc = eigVc.T
    current = np.sum((eigVc.T.conj().dot(Derivative.dot(eigVc)) / eigVl)
                     for eigVl, eigVc in zip(eigVl, eigVc))
    current = -0.5 * Delta ** 2 * current.real
    energy = -np.sum(eigVl)
    
#     #proba wyliczenia energii inaczej: ale kształt wychodzi taki sam
    M11 = np.zeros((N + M, N + M))
    M = np.bmat([[M11, -1j * A.T.conj()], [1j * A, M11]])
    eigVl, _ = la.eigh(M) 
    eigVl = eigVl * Delta
    energy = -np.sum(np.absolute(eigVl[eigVl<0]))
    return current, energy, eigVl

# ============================================================================
# Supercurrent density maps
# ============================================================================


def andreev_states(syst, par, phi, Delta):
    """Returns Andreev eigenvalues and eigenvectors.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem object
        A finalized kwant system having a scattering region
        connected with two semi-infinite leads.
    par : SimpleNamespace object
        Simplenamespace object with Hamiltonian parameters.
    phi : float
        Superconducting phase difference between the two superconducting leads.
    Delta : float
        Superconducting gap.
    """
    s = kwant.smatrix(syst, energy=0, args=[par])
    N, M = [len(li.momenta) // 2 for li in s.lead_info]
    s = s.data
    r_a11 = 1j * np.eye(N)
    r_a12 = np.zeros((N, M))
    r_a21 = r_a12.T
    r_a22 = 1j * np.exp(- 1j * phi) * np.eye(M)
    r_a = np.bmat([[r_a11, r_a12], [r_a21, r_a22]])
    zeros = np.zeros(shape=(len(s), len(s)))
    matrix = np.bmat([[zeros, (s.T.conj()).dot(r_a.conj())],
                      [(s.T).dot(r_a), zeros]])
    eigVl, eigVc = la.eig(matrix)
    eigVc = la.qr(eigVc)[0]
    eigVl = eigVl * Delta
    values = []
    vectors = []
    for ii in range(len(eigVl)):
        if eigVl[ii].real > 0 and eigVl[ii].imag > 0:
            values.append(eigVl[ii].real)
            vectors.append(eigVc.T[ii][0:len(eigVl) // 2])
    values = np.array(values)
    vectors = np.array(vectors)
    return values, vectors


def andreev_wf(eigvec, kwant_wf):
    """
    Returns Andreev wavefunctions using eigenvalues and eigenvectors from
    the bound-state eigenvalue problem.

    Parameters
    ----------
    eigvec : numpy array
        Eigenvectors from the Andreev bound-state condition.
    kwant_wf : kwant.solvers.common.WaveFunction object
        Wavefunctions of a normal scattering region connected
        with two normal leads.
    """
    w = np.vstack((kwant_wf(0), kwant_wf(1)))
    and_wf = [np.dot(vec, w) for vec in eigvec]
    return and_wf


def intensity(syst, psi, par):
    """Returns the current through a kwant system.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem object
        A finalized kwant system having a scattering region connected
        with two semi-infinite leads.
    psi : numpy array
        Andreev wavefunctions constructed from kwant wavefunctions and
        Andreev bound-state eigenvalue problem.
    par : SimpleNamespace object
        Simplenamespace object with Hamiltonian parameters.
    """
    I_operator = kwant.operator.Current(syst)
    return sum(I_operator(psi_i, args=[par]) for psi_i in psi)