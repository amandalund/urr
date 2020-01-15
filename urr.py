from math import sqrt, log, pi
from random import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openmc
from openmc.data.reconstruct import penetration_shift, wave_number


def ladder(urr):
    """Generate a resonance ladder spanning the entire unresolved resonance
    range.

    Parameters
    ----------
    urr : openmc.data.Unresolved
        Unresolved resonance parameters

    Returns
    -------
    list of tuple
        Sampled resonance ladder

    """
    lad = []
    ljs = []
    params = urr.parameters

    # Keep ~300 resonances on either end of the unresolved resonance range to
    # avoid truncation effects
    d_min = params['d'].iloc[0]
    d_max = params['d'].iloc[-1]
    energy_min = max(urr.energy_min - 300 * d_min, 1e-1)
    energy_max = urr.energy_max + 300 * d_max

    if 'amux' in params.columns:
        case = 'C'
    elif 'amuf' in params.columns:
        case = 'B'
    else:
        case = 'A'

    def expand_row_params(idx):
        row = params.iloc[idx]
        if case == 'A':
            # Case A fission widths not given, all parameters energy-independent
            l, j, d, amun, gn0, gg = row
            E = gf = gx = amux = amuf = 0.0
        elif case == 'B':
            # Case B: fission widths given, only fission widths energy-dependent
            l, j, E, d, amun, amuf, gn0, gg, gf = row
            gx = amux = 0.0
        else:
            # Case C: fission widths given, all parameters energy-dependent
            l, j, E, d, amux, amun, amuf, gx, gn0, gg, gf = row
        return l, j, E, d, amux, amun, amuf, gx, gn0, gg, gf

    for row in params.itertuples():
        # Do attribute access up front
        l, j, E_l, d_l, amux_l, amun_l, amuf_l, gx_l, gn0_l, gg_l, gf_l = expand_row_params(row.Index)

        # New spin sequence (l, j)
        if (l, j) not in ljs:
            ljs.append((l, j))

            # Select a starting energy for this spin sequence
            energy = energy_min + random() * d_l

        if urr.energies:
            if energy < urr.energy_min or E_l >= urr.energy_max:
                E_r, d_r, amux_r, amun_r, amuf_r, gx_r, gn0_r, gg_r, gf_r = expand_row_params(row.Index)[2:]
            else:
                E_r, d_r, amux_r, amun_r, amuf_r, gx_r, gn0_r, gg_r, gf_r = expand_row_params(row.Index + 1)[2:]

        if case == 'A':
            # Get the parameters for this spin sequence if they are not
            # energy-dependent (case A). There is no competitive width.
            avg_d = d_l
            avg_amun = amun_l
            avg_gn0 = gn0_l
            avg_gg = gg_l
            avg_gf = 0
            avg_gx = 0

        while energy < energy_max:
            # Interpolate energy-dependent parameters
            if urr.energies:
                if E_r == E_l:
                    f = 0
                else:
                    f = (energy - E_l)/(E_r - E_l)
                avg_d = d_l + f*(d_r - d_l)
                avg_amun = amun_l + f*(amun_r - amun_l)
                avg_amuf = amuf_l + f*(amuf_r - amuf_l)
                avg_amux = amux_l + f*(amux_r - amux_l)
                avg_gn0 = gn0_l + f*(gn0_r - gn0_l)
                avg_gg = gg_l + f*(gg_r - gg_l)
                avg_gf = gf_l + f*(gf_r - gf_l)
                avg_gx = gx_l + f*(gx_r - gx_l)

            # Sample fission width
            if avg_gf == 0:
                gf = 0
            else:
                xf = np.random.chisquare(avg_amuf)
                gf = xf*avg_gf/avg_amuf

            # Sample competitive width
            if avg_gx == 0:
                gx = 0
            else:
                xx = np.random.chisquare(avg_amux)
                gx = xx*avg_gx/avg_amux

            # Calculate energy-dependent neutron width
            xn0 = np.random.chisquare(avg_amun)
            k = wave_number(urr.atomic_weight_ratio, energy)
            rho = k*urr.channel_radius(energy)
            p_l, _ = penetration_shift(l, rho)
            gn = p_l/rho*sqrt(energy)*xn0*avg_gn0

            # Calculate total width
            gt = gn + avg_gg + gf + gx

            # Sample level spacing
            d = avg_d*sqrt(-4*log(random())/pi)

            # Update resonance parameters and energy
            lad.append((energy, l, j, gt, gn, avg_gg, gf, gx))
            energy += d

            # If the parameters are energy-dependent (Case C) or fission widths are
            # energy-dependent (Case B), get the parameters for the next energy bin
            # for this spin sequence
            if urr.energies and energy > E_r and E_l < urr.energy_max:
                break

    return lad


def sample_xs(urr, energy, num_resonances):
    """Generate a realization of cross section values.

    Parameters
    ----------
    urr : openmc.data.Unresolved
        Unresolved resonance parameters
    energy : float
        Incident neutron energy
    num_resonances : float
        Number of resonances contributing to the calculation of the cross
        section at the given energy

    Returns
    -------
    dict of int to float
        Sampled cross section values for each reaction type

    """
    # Stochastically generate a resonance ladder
    lad = ladder(urr)
    columns = ['energy', 'L', 'J', 'totalWidth', 'neutronWidth',
               'captureWidth', 'fissionWidth', 'competitiveWidth']
    df = pd.DataFrame.from_records(lad, columns=columns)

    # Find the indices in the dataframe that will give num_resonances around
    # the energy for each spin sequence
    indices = []
    for L in df.L.unique():
        for J in df[df.L == L].J.unique():
            i = df.index[(df.L == L) & (df.J == J) & (df.energy >= energy)][0]
            indices.extend(list(range(i - num_resonances//2, i + num_resonances//2)))
    parameters = df.iloc[indices].reset_index(drop=True)

    # Set channel radius, scattering radius, and Q-value
    l_values = parameters.L.unique()
    channel_radius = {l: urr.channel_radius for l in l_values}
    scattering_radius = {l: urr.scattering_radius for l in l_values}
    q_value = {l: 0 for l in l_values}

    # Create a "resolved" range
    rr = openmc.data.SingleLevelBreitWigner(
        urr.target_spin,
        urr.energy_min,
        urr.energy_max,
        channel_radius,
        scattering_radius
    )
    rr.parameters = parameters
    rr.atomic_weight_ratio = urr.atomic_weight_ratio
    rr.q_value = q_value

    # Compute cross sections
    xs = rr.reconstruct(energy)
    sampled_xs = {2: xs[2], 102: xs[102], 18: xs[18]}
    sampled_xs[1] = xs[2] + xs[102] + xs[18]

    return sampled_xs


if __name__ == "__main__":
    # Case A: fission widths are not given; all parameters are energy-independent
    case_a = '/Users/amandalund/openmc/data/endf-b-vii.1/neutrons/n-068_Er_167.endf'

    # Case B: fission widths are given; only fission widths are energy-dependent
    case_b = '/Users/amandalund/openmc/data/endf-b-vii.1/neutrons/n-094_Pu_240.endf'

    # Case C: all parameters are energy-dependent
    case_c = '/Users/amandalund/openmc/data/endf-b-vii.1/neutrons/n-092_U_238.endf'

    nuc = openmc.data.IncidentNeutron.from_endf(case_c)
    urr = nuc.resonances.unresolved

    sampled_xs = {2: [], 102: [], 18: [], 5: [], 1: []}
    num_realizations = 10
    num_resonances = 100
    energy = 50.0e3

    # Sample some cross section values (this will be slow since we are generating
    # resonance ladders across the entire unresolved energy range)
    for _ in range(num_realizations):
        xs = sample_xs(urr, energy, num_resonances)
        for mt in 2, 102, 18:
            sampled_xs[mt].append(xs[mt])
        sampled_xs[1].append(xs[2] + xs[102] + xs[18])

    # Plot a histogram of the sampled cross sections
    fig = plt.figure(figsize=(8,6), facecolor='w')
    h = plt.hist(sampled_xs[2], bins=50, color='k')
    plt.grid(True, alpha=0.5)
    plt.show()
