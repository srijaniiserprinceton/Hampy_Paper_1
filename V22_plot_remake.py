import cdflib, os, re, xarray
import numpy as np
import matplotlib.pyplot as plt; plt.ion(); plt.rcParams['font.size'] = 16

import encounters

def get_file_indices_from_enc(start_enc, end_enc, num_days_around_perihelion, hamstring_dir):
    enc_days = []

    for enc in range(start_enc, end_enc+1):
        if(enc < 10): enc_str = f'E0{enc}'
        else: enc_str = f'E{enc}'
        pday = encounters.get_enc_dates(enc_str).astype('datetime64[D]')

        this_enc_days = np.arange(pday - np.timedelta64(num_days_around_perihelion, 'D'),
                                  pday + np.timedelta64(num_days_around_perihelion+1, 'D'))
        
        enc_days.append(this_enc_days)
    
    enc_days = np.asarray(enc_days).flatten()

    # finding which file indices in the hamstring folder falls within these dates
    filenames = os.listdir(hamstring_dir)

    # extracting the dates for each file
    filedates = np.asarray([np.datetime64(re.split('[_]', f)[1]).astype('datetime64[D]') for f in filenames])

    file_indices = np.asarray([(filedates[i] in enc_days) for i in range(len(filedates))])

    return file_indices, filedates

def plot_Taniham_fracden(ax, hamdata, cmap='turbo'):
    Tani_ham = hamdata['Tperp_ham'].data / hamdata['Tpar_ham'].data
    nham_ntot = hamdata['n_ham'].data / (hamdata['n_ham'].data + hamdata['n_neck'].data + hamdata['n_core'].data)

    nanmask = np.isnan(Tani_ham) | np.isnan(nham_ntot)
    good_indices = ~nanmask

    ax.hist2d(np.log10(nham_ntot[good_indices]), Tani_ham[good_indices], range=[[-5,0],[0,10]], bins=[50,50], density=True, vmax=0.1, cmap=cmap, rasterized=True)

def plot_Taniham_fracvel(ax, hamdata, cmap='turbo'):
    Tani_ham = hamdata['Tperp_ham'].data / hamdata['Tpar_ham'].data

    rho_tot = hamdata['n_core'] + hamdata['n_neck'] + hamdata['n_ham']
    Bmag = np.sqrt(hamdata['Bx_inst']**2 + hamdata['By_inst']**2 + hamdata['Bz_inst']**2)
    vA = 21.8 * Bmag / np.sqrt(rho_tot)
    vdrift = hamdata['vx_inst_ham'] - hamdata['vx_inst_core']

    fracvel = vdrift / vA

    nanmask = np.isnan(Tani_ham) | np.isnan(fracvel)
    good_indices = ~nanmask

    ax.hist2d(fracvel[good_indices], Tani_ham[good_indices], range=[[-6,0],[0,10]], bins=[50,50], density=True, vmax=0.2, cmap=cmap, rasterized=True)

if __name__=='__main__':
    start_enc, end_enc = 4, 23
    num_days_around_perihelion = 10

    # extracting which files should be loaded corresponding to the desired encounter
    file_indices, filedates = get_file_indices_from_enc(start_enc, end_enc, num_days_around_perihelion, 'Hamstrings')

    filenames = np.sort(filedates[file_indices])

    xrs = []

    for f in filenames:
        xr = cdflib.cdf_to_xarray(f'Hamstrings/hamstring_{f}_v02.cdf')
        xrs.append(xr)

    hamdata = xarray.concat(xrs, dim='record0')
    eventmask = 

    # making the plot
    fig, ax = plt.subplots(1, 2, figsize=(12,4), sharey=True)
    plot_Taniham_fracden(ax[0], hamdata, eventmask)
    plot_Taniham_fracvel(ax[1], hamdata, eventmask)

    ax[0].set_ylabel(r'$T^{\mathrm{ham}}_{\perp} / T^{\mathrm{ham}}_{\parallel}$')
    ax[0].set_xlabel(r'$\log_{10}\left(n^{\mathrm{ham}} / n^{\mathrm{total}}\right)$')
    ax[1].set_xlabel(r'$v^{\mathrm{ham}}_{\mathrm{drift}} / v_A$')

    plt.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.97, wspace=0.1)

    plt.savefig('V22_replot_allEnc.pdf')