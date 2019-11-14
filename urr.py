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
    
    for row in urr.parameters.itertuples():
        # New spin sequence (l, j)
        if (row.L, row.J) not in ljs:
            ljs.append((row.L, row.J))
            
            # Select a starting energy for this spin sequence
            energy = urr.energy_min + random() * row.d
            
        if urr.energies and row.E < urr.energy_max:
            # Get the next row to interpolate energy-dependent parameters
            row1 = urr.parameters.iloc[row.Index + 1]
        else:
            # Get the parameters for this spin sequence if they are not
            # energy-dependent (case A). There is no competitive width.
            avg_d = row.d
            avg_amun = row.amun
            avg_gn0 = row.gn0
            avg_gg = row.gg
            avg_gf = 0
            avg_gx = 0
            
        while energy < urr.energy_max:
            # Interpolate energy-dependent parameters
            if urr.energies:
                f = (energy - row.E)/(row1.E - row.E)
                avg_d = row.d + f*(row1.d - row.d)
                avg_amun = row.amun + f*(row1.amun - row.amun)
                avg_amuf = row.amuf + f*(row1.amuf - row.amuf)
                if 'amux' in urr.parameters.columns:
                    avg_amux = row.amux + f*(row1.amux - row.amux)
                avg_gn0 = row.gn0 + f*(row1.gn0 - row.gn0)
                avg_gg = row.gg + f*(row1.gg - row.gg)
                avg_gf = row.gf + f*(row1.gf - row.gf)
                if 'gx' in urr.parameters.columns:
                    avg_gx = row.gx + f*(row1.gx - row.gx)
                else:
                    avg_gx = 0
            
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
            p_l, s_l = penetration_shift(row.L, rho)
            gn = p_l/rho*sqrt(energy)*xn0*avg_gn0

            # Calculate total width
            gt = gn + avg_gg + gf + gx
            
            # Sample level spacing
            d = avg_d*sqrt(-4*log(random())/pi)
            
            # Update resonance parameters and energy
            lad.append((energy, row.L, row.J, gt, gn, avg_gg, gf, gx))
            energy += d            
        
            # If the parameters are energy-dependent (Case C) or fission widths are
            # energy-dependent (Case B), get the parameters for the next energy bin
            # for this spin sequence
            if urr.energies and energy > row1.E:
                continue

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
