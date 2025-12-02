import cdflib, os, re, xarray, pickle
import numpy as np
import matplotlib.pyplot as plt; plt.ion(); plt.rcParams['font.size'] = 12
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import encounters
import misc_functions as misc_funcs

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

def plot_Taniham_fracden(ax, hamdata, cmap='magma'):
    Tani_ham = hamdata['Tperp_ham'].data / hamdata['Tpar_ham'].data
    nham_ntot = hamdata['n_ham'].data / (hamdata['n_ham'].data + hamdata['n_neck'].data + hamdata['n_core'].data)

    nanmask = np.isnan(Tani_ham) | np.isnan(nham_ntot)
    good_indices = ~nanmask

    H, xedges, yedges, im = ax.hist2d(np.log10(nham_ntot[good_indices]), Tani_ham[good_indices], range=[[-5,0],[0,10]], bins=[50,50], density=True, vmax=0.17, cmap=cmap, rasterized=True)
    return H, xedges, yedges, im

def plot_Taniham_fracvel(ax, hamdata, cmap='magma'):
    Tani_ham = hamdata['Tperp_ham'].data / hamdata['Tpar_ham'].data

    rho_tot = hamdata['n_core'] + hamdata['n_neck'] + hamdata['n_ham']
    Bmag = np.sqrt(hamdata['Bx_inst']**2 + hamdata['By_inst']**2 + hamdata['Bz_inst']**2)
    vA = 21.8 * Bmag / np.sqrt(rho_tot)
    vdrift = hamdata['vx_inst_ham'] - hamdata['vx_inst_core']

    fracvel = vdrift / vA

    nanmask = np.isnan(Tani_ham) | np.isnan(fracvel)
    good_indices = ~nanmask

    H, xedges, yedges, im = ax.hist2d(fracvel[good_indices], Tani_ham[good_indices], range=[[-6,0],[0,10]], bins=[50,50], density=True, vmax=0.2, cmap=cmap, rasterized=True)
    return H, xedges, yedges, im

def put_enc_textbox(ax_left, ax_right, enc_str, nhams):
    # Get bounding boxes in figure coordinates
    bbox_L = ax_left.get_position()
    bbox_R = ax_right.get_position()

    # Compute midpoint between the two subplots
    xmid = 0.5 * (bbox_L.x1 + bbox_R.x0)
    ymid = bbox_L.y1 - 0.01   # small offset above the top of the mini-panels

    # Add textbox using figure coordinates
    fig.text(
        xmid, ymid,
        f'{enc_str} | #{nhams}',                       # put your label here
        ha='center', va='bottom',
        fontsize=10,
        bbox=dict(
            facecolor='white',
            alpha=0.9,
            boxstyle='round,pad=0.3',
            edgecolor='black'
        )
    )

def write_pickle(x, fname):
    with open(f'{fname}.pkl', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    start_enc, end_enc = 4, 23
    num_days_around_perihelion = 10

    ncases = 20
    nrows, ncols = 10, 2

    fig = plt.figure(figsize=(12, 16))
    outer_grid = gridspec.GridSpec(nrows, ncols, wspace=0.1, hspace=0.2, left=0.08, right=0.99, top=0.99, bottom=0.06)

    for enc_idx, enc in enumerate(np.arange(start_enc, end_enc+1)):
        # extracting which files should be loaded corresponding to the desired encounter
        file_indices, filedates = get_file_indices_from_enc(enc, enc, num_days_around_perihelion, 'Hamstrings')

        filenames = np.sort(filedates[file_indices])

        xrs = []

        for f in filenames:
            xr = cdflib.cdf_to_xarray(f'Hamstrings/hamstring_{f}_v02.cdf')
            xrs.append(xr)

        hamdata = xarray.concat(xrs, dim='record0')

        # converting the epoch to datetime for the detected hammerhead times
        hammertimes = cdflib.cdfepoch.to_datetime(hamdata['epoch'].data)

        # sys.exit()

        # getting the bins in time as per angular distance
        bins, bin_idx, ham_frac_in_bin, ham_counts_in_bin, all_counts_in_bin =\
                        misc_funcs.get_angular_bins_in_encounter(hammertimes, 'Hamstrings', enc,
                                                                 angular_sep_deg=1,
                                                                 spice_cadence_days=1/(12*24),
                                                                 plot_trajectory=True)
        # bins, bin_idx =\
        #                 misc_funcs.get_angular_bins_in_encounter(hammertimes, 'Hamstrings', enc,
        #                                                          angular_sep_deg=0.1,
        #                                                          spice_cadence_days=1/(12*24),
        #                                                          plot_trajectory=True)

        # ham_bins = misc_funcs.count_points_in_bins(hammertimes, bins)
        continue

        # making the plot
        # Locate which big panel we are in
        row = enc_idx % nrows
        col = enc_idx // nrows

        # Create a 1×2 grid *inside* the panel
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            1, 2,
            subplot_spec=outer_grid[row, col],
            wspace=0.05
        )

        ax_left  = fig.add_subplot(inner_grid[0, 0])
        ax_right = fig.add_subplot(inner_grid[0, 1])

        H, xedges, yedges, im = plot_Taniham_fracden(ax_left, hamdata)
        if(enc_idx == 0):
            H0_L = H
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            X_L, Y_L = np.meshgrid(xc, yc)

            norm = im.norm          # this is the Normalize / LogNorm used
            frac = 0.8               # 0–1 along the colorbar
            vmin, vmax = norm.vmin, norm.vmax

            if hasattr(norm, 'log10'):  # crude check for LogNorm
                level_L = vmin * (vmax / vmin) ** frac
            else:                        # Linear Normalize
                level_L = vmin + frac * (vmax - vmin)
        ax_left.contour(X_L, Y_L, H0_L.T, levels=[level_L], colors='white', linewidths=2.0)

        H, xedges, yedges, im = plot_Taniham_fracvel(ax_right, hamdata)
        if(enc_idx == 0):
            H0_R = H
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            X_R, Y_R = np.meshgrid(xc, yc)

            norm = im.norm          # this is the Normalize / LogNorm used
            frac = 0.8               # 0–1 along the colorbar
            vmin, vmax = norm.vmin, norm.vmax

            if hasattr(norm, 'log10'):  # crude check for LogNorm
                level_R = vmin * (vmax / vmin) ** frac
            else:                        # Linear Normalize
                level_R = vmin + frac * (vmax - vmin)
        ax_right.contour(X_R, Y_R, H0_R.T, levels=[level_R], colors='white', linewidths=2.0)

        ax_left.tick_params(which="both", labelbottom=False, labelleft=False)
        ax_right.tick_params(which="both", labelbottom=False, labelleft=False)

        # Leftmost column → left panels get y–labels
        if col == 0:
            ax_left.tick_params(labelleft=True)
            ax_left.set_ylabel(r'$T^{\mathrm{ham}}_{\perp} / T^{\mathrm{ham}}_{\parallel}$')

        # Bottom row → x–labels on both mini-panels
        if row == nrows - 1:
            ax_left.set_xticks([])
            ax_left.set_xticks([-4, -2, 0])
            ax_left.tick_params(labelbottom=True)
            ax_left.set_xlabel(r'$\log_{10}\left(n^{\mathrm{ham}} / n^{\mathrm{total}}\right)$')

            ax_right.set_xticks([])
            ax_right.set_xticks([-5, -2.5, 0])
            ax_right.tick_params(labelbottom=True)
            ax_right.set_xlabel(r'$v^{\mathrm{ham}}_{\mathrm{drift}} / v_A$')

        if(enc < 10): enc_str = f'E0{enc}'
        else: enc_str = f'E{enc}'
        put_enc_textbox(ax_left, ax_right, enc_str, len(hamdata['epoch'].data))

    plt.savefig('V22_replot_eachEnc.pdf')