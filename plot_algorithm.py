import cdflib, bisect, sys, re
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt; plt.ion(); plt.rcParams['font.size'] = 16

from hampy import get_span_data as get_data
from hampy import nonjax_functions as f
from hampy import comp_moments

# global variables
mass_p = 0.010438870      #proton mass in units eV/c^2 where c = 299792 km/s
charge_p = 1              #proton charge in units eV

NAX = np.newaxis

def get_edges(r, theta, phi_const):
    r_edges = 0.5 * (r[1:] + r[:-1])
    r_edges = np.concatenate(([r[0] - (r_edges[0]-r[0])], r_edges, [r[-1] + (r[-1]-r_edges[-1])]))

    theta_edges = 0.5 * (theta[1:] + theta[:-1])
    theta_edges = np.concatenate(([theta[0] - (theta_edges[0]-theta[0])], theta_edges, [theta[-1] + (theta[-1]-theta_edges[-1])]))

    # Make mesh of edges
    R_edges, T_edges = np.meshgrid(r_edges, theta_edges, indexing='ij')
    vx_edges = R_edges * np.cos(phi_const) * np.cos(T_edges)
    vz_edges = R_edges * np.sin(T_edges)

    return vx_edges, vz_edges

def plot_preprocessing(log_df_theta_pre, log_df_theta_post, vx, vz, r, theta, phi_const, cmap='plasma', alpha=0.9):
    # drawing the straight lines for demonstrating the 1D convolution exercise
    theta_top = theta[1]
    theta_bottom = theta[-3]

    # making the straight lines
    x = np.linspace(-920, 0, 100)
    y = np.zeros((len(theta), len(x)))
    
    for thetaidx, thetaval in enumerate(theta):
        y[thetaidx] = np.tan(thetaval) * x

    # making the subplot
    fig, ax = plt.subplots(1, 2, figsize=(7, 6), sharey=True)

    # computing the edges grid
    vx_edges, vz_edges = get_edges(r, theta, phi_const)

    vmin, vmax = np.nanmin(log_df_theta_post), np.nanmax(log_df_theta_post)

    # plotting the pre-processed df
    im = ax[0].pcolormesh(vx, vz, log_df_theta_pre, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, rasterized=True)
    ax[0].set_xlabel(r'$V_x$ [km/s]')
    ax[0].set_ylabel(r'$V_z$ [km/s]')
    ax[0].text(0.9, 0.97, '(a)', transform=ax[0].transAxes,
               horizontalalignment='center', verticalalignment='top',
               fontsize=16, fontweight='bold')

    # plotting the post-processed df
    ax[1].pcolormesh(vx, vz, log_df_theta_post, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, rasterized=True)
    ax[1].set_xlabel(r'$V_x$ [km/s]')
    ax[1].text(0.9, 0.97, '(b)', transform=ax[1].transAxes,
               horizontalalignment='center', verticalalignment='top',
               fontsize=16, fontweight='bold')

    [ax[i].set_aspect('equal') for i in range(2)]
    [ax[i].set_xlim([-920,None]) for i in range(2)]
    [ax[i].set_ylim([-800,800]) for i in range(2)]

    for j in range(vx_edges.shape[1]):
        ax[0].plot(vx_edges[:, j], vz_edges[:, j], color='k', lw=0.5, alpha=0.3, zorder=0)
        ax[1].plot(vx_edges[:, j], vz_edges[:, j], color='k', lw=0.5, alpha=0.3, zorder=0)
    for i in range(vx_edges.shape[0]):
        ax[0].plot(vx_edges[i, :], vz_edges[i, :], color='k', lw=0.5, alpha=0.3, zorder=0)
        ax[1].plot(vx_edges[i, :], vz_edges[i, :], color='k', lw=0.5, alpha=0.3, zorder=0)

    # ax[1].plot(x, y1, '--k', lw=3)
    # ax[1].plot(x, y2, '--k', lw=3)

    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.13, right=0.98, wspace=0.04)
    cbar_ax = fig.add_axes([0.2, 0.92, 0.7, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar_ax.set_title(r'$log_{10}(f)$')
    plt.savefig('Fig1ab.pdf')
    
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 6), sharey=True)
    im = ax.contourf(vx, vz, log_df_theta_pre, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, rasterized=True)
    ax.set_xlabel(r'$V_x$ [km/s]')
    ax.set_ylabel(r'$V_z$ [km/s]')
    ax.text(0.9, 0.97, '(c)', transform=ax.transAxes,
               horizontalalignment='center', verticalalignment='top',
               fontsize=16, fontweight='bold')

    for j in range(vx_edges.shape[1]):
        ax.plot(vx_edges[:, j], vz_edges[:, j], color='k', lw=0.5, alpha=0.3, zorder=0)
    for i in range(vx_edges.shape[0]):
        ax.plot(vx_edges[i, :], vz_edges[i, :], color='k', lw=0.5, alpha=0.3, zorder=0)

    ax.set_aspect('equal')
    ax.set_xlim([-600,-100])
    ax.set_ylim([-440,440])

    for i in range(len(theta)):
        if(i == 1):
            plt.plot(x, y[i], lw=2, color='r')
        elif(i == 5):
            plt.plot(x, y[i], lw=2, color='r', ls='--')
        else:
            plt.plot(x, y[i], lw=2, color='k')
        angle = np.degrees(np.arctan(y[i,10]/x[10]))
        plt.text(x[-37], y[i,-37],
                 r"$\theta = %i^{\circ}$"%np.degrees(theta[i]), color='k', fontsize=12, rotation=angle,
                 rotation_mode='anchor', ha='left', va='bottom', fontweight='bold')

    fig.subplots_adjust(top=0.98, bottom=0.11, left=0.17, right=0.99, wspace=0.04)
    plt.savefig('Fig1c.pdf')

def plot_gap_identification(vx, vz, v, log_df_theta_post):
    # calculating the gaps
    # 1. creating the 1D and 2D convolution matrices once for the entire runtime
    convmat = f.convolve_hammergap(vx, vz)
    # 2. performing the 1D and 2D convolutions and recording the results
    convmat.conv1d_w_VDF(log_df_theta_post)
    convmat.conv2d_w_VDF(log_df_theta_post)
    # 3. finding if there are gaps symmetric about the theta for max vdf (core)
    convmat.merge_1D_2D(np.where(log_df_theta_post == np.nanmax(log_df_theta_post))[0][0])

    # making the mask of the non-zero parts of the post-processed VDF
    log_VDF = np.nan_to_num(log_df_theta_post * 1.0)
    mask_vdf = log_VDF <= np.nanmin(log_VDF[log_VDF > 0])

    # plotting the gaps with VDFs overlayed
    fig, ax = plt.subplots(1, 2, figsize=(7, 6), sharey=True)

    # making the modified colorbar
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=256):
        cmap = cm.get_cmap(cmap_name, n)
        new_colors = cmap(np.linspace(minval, maxval, n))
        new_cmap = mcolors.ListedColormap(new_colors)
        return new_cmap

    cmap_trimmed = truncate_colormap('gist_stern_r', 0.0, 0.95)

    vmin, vmax = np.nanmin(log_df_theta_post), np.nanmax(log_df_theta_post)

    # filling up the 2-gap convolution colormap
    hammermat_2 = np.flip(convmat.gap_mat_1D['2'], axis=0)
    color_conv_2 = np.zeros_like(mask_vdf, dtype='float64')
    for i in range(8): color_conv_2[i] = np.convolve(mask_vdf[i], hammermat_2, mode='same')

    # filling up the 3-gap convolution colormap
    hammermat_3 = np.flip(convmat.gap_mat_1D['3'], axis=0)
    color_conv_3 = np.zeros_like(mask_vdf, dtype='float64')
    for i in range(8): color_conv_3[i] = np.convolve(mask_vdf[i], hammermat_3, mode='same')

    im1=ax[0].pcolormesh(vx_plane_theta, vz_plane_theta, color_conv_2.T, vmin=-2, vmax=2, cmap=cmap_trimmed, rasterized=True)
    ax[0].contourf(vx, vz, log_df_theta_post.T, vmin=vmin, vmax=vmax, cmap='binary_r', alpha=0.8, rasterized=True)
    im2=ax[1].pcolormesh(vx_plane_theta, vz_plane_theta, color_conv_3.T, vmin=-3, vmax=3, cmap=cmap_trimmed, rasterized=True)
    ax[1].contourf(vx, vz, log_df_theta_post.T, vmin=vmin, vmax=vmax, cmap='binary_r', alpha=0.8, rasterized=True)

    for a in ax:
        a.set_aspect('equal')
        a.set_xlim([-600,-100])
        a.set_ylim([-440,440])

    cbar_ax1 = fig.add_axes([0.13, 0.95, 0.37, 0.02])
    cb1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
    cb1.set_ticks([-2, 0, 2])
    # cbar_ax1.set_title(r'$log_{10}(f)$')
    cbar_ax2 = fig.add_axes([0.58, 0.95, 0.37, 0.02])
    cb2 = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
    cb2.set_ticks([-3, 0, 3])
    # cbar_ax1.set_title(r'$log_{10}(f)$')

    ax[0].set_xlabel(r'$V_x$ [km/s]')
    ax[1].set_xlabel(r'$V_x$ [km/s]')
    ax[0].set_ylabel(r'$V_z$ [km/s]')

    ax[0].text(0.9, 0.97, '(e)', transform=ax[0].transAxes,
               horizontalalignment='center', verticalalignment='top',
               fontsize=16, fontweight='bold')
    ax[1].text(0.9, 0.97, '(f)', transform=ax[1].transAxes,
               horizontalalignment='center', verticalalignment='top',
               fontsize=16, fontweight='bold')

    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.12, right=0.99, wspace=0.04)
    plt.savefig('Fig1ef.pdf')

    # making the 1D convolution detection plot
    fig, ax = plt.subplots(2, 1, figsize=(4,8), sharex=True)

    for i in range(8):
        if(i==6):
            ax[0].plot(v, color_conv_2.T[:,i], 'r', lw=2, label=r'$\theta = -37^{\circ}$')
        else:
            ax[0].plot(v, color_conv_2.T[:,i], 'k', lw=2, alpha=0.5)
    ax[0].legend(loc='lower right', framealpha=0.8)   
    ax[0].grid(True)

    for i in range(8):
        if(i==2):
            ax[1].plot(v, color_conv_3.T[:,i], '--r', lw=2, label=r'$\theta = 22^{\circ}$')
        else:
            ax[1].plot(v, color_conv_3.T[:,i], 'k', lw=2, alpha=0.5)
    ax[1].legend(loc='lower right', framealpha=0.8) 
    ax[1].grid(True)

    [ax[i].set_xlim([100,600]) for i in range(2)]

    ax[1].set_xlabel('velocity [km/s]')
    ax[0].set_title('1D convolution')

    ax[0].text(0.9, 0.97, '(d1)', transform=ax[0].transAxes,
               horizontalalignment='center', verticalalignment='top',
               fontsize=16, fontweight='bold')
    ax[1].text(0.9, 0.97, '(d2)', transform=ax[1].transAxes,
               horizontalalignment='center', verticalalignment='top',
               fontsize=16, fontweight='bold')

    fig.subplots_adjust(top=0.95, bottom=0.10, left=0.12, right=0.95, wspace=0.04, hspace=0.1)
    plt.savefig('Fig1d.pdf')

'''
def plot_preprocessing_method(df_theta, cmap='binary_r'):
    log_df_theta = np.nan_to_num(np.log10(df_theta), nan=np.nan, posinf=np.nan, neginf=np.nan)

    # filtering to throw out pixels which dont have a finite value on an adjacent (not diagonal) cell
    log_df_theta_padded = np.zeros((34, 10)) + np.nan
    log_df_theta_padded[1:-1, 1:-1] = log_df_theta

    filter_mask = np.zeros_like(log_df_theta, dtype='bool')

    fig, ax = plt.subplots(2, 2, figsize=(10,10) ,sharex=True, sharey=True)

    filter_mask += ~np.isnan(log_df_theta_padded[0:-2,1:-1])
    ax[0,0].pcolormesh(~filter_mask, cmap=cmap, alpha=0.3)

    filter_mask += ~np.isnan(log_df_theta_padded[2:,1:-1])
    ax[0,1].pcolormesh(~filter_mask, cmap=cmap, alpha=0.3)

    filter_mask += ~np.isnan(log_df_theta_padded[1:-1,0:-2])
    ax[1,0].pcolormesh(~filter_mask, cmap=cmap, alpha=0.3)

    filter_mask += ~np.isnan(log_df_theta_padded[1:-1,2:])
    ax[1,1].pcolormesh(~filter_mask, cmap=cmap, alpha=0.3)

    [axs.pcolormesh(log_df_theta, cmap='plasma', alpha=0.5) for axs in ax.flatten()[:-1]]

    log_df_theta_new = log_df_theta * 1.0
    log_df_theta_new[~filter_mask] = np.nan

    plt.savefig('Pre-processing.pdf')
'''

if __name__=='__main__':
    # timestamp for Verniero et al 2022 hammerhead
    year, month, date = 2020, 1, 29
    hour, minute, second = 18, 10, 2

    # converting to datetime format to extract time index
    user_datetime = datetime(year, month, date)
    timeSlice  = np.datetime64(datetime(year, month, date, hour, minute, second))

    # loading the data (downloading the file if necessary)
    cdf_VDfile = get_data.download_VDF_file(user_datetime)
    # getting the spi_vars 
    l3_data = get_data.download_L3_data(user_datetime)

    # convert time
    epoch = cdflib.cdfepoch.to_datetime(cdf_VDfile['EPOCH'])
    # find index for desired timeslice
    tSliceIndex  = bisect.bisect_left(epoch, timeSlice)
    
    # getting the VDF dictionary at the desired timestamp
    vdf_dict = get_data.get_VDFdict_at_t(cdf_VDfile, tSliceIndex)

    # convert time
    epoch = cdflib.cdfepoch.to_datetime(l3_data['EPOCH'])
    # find index for desired timeslice
    tSliceIndex  = bisect.bisect_left(epoch, timeSlice)
    # getting the required l3 data dictionary
    l3_data_dict = get_data.get_L3_monents_at_t(l3_data, tSliceIndex)

    # updating the vdf_dict with the required L3 data
    vdf_dict.update(l3_data_dict)

    # DATA FORMAT OF THE VDF: phi is along dimension 0, while theta is along 2
    # choosing a cut through phi for plotting
    phi_cut=1 

    phi_plane = vdf_dict['phi'][phi_cut,:,:]
    theta_plane = vdf_dict['theta'][phi_cut,:,:]
    energy_plane = vdf_dict['energy'][phi_cut,:,:]
    vel_plane = np.sqrt(2 * charge_p * energy_plane / mass_p)

    # VDF as a function of energy and theta (phi axis is summed over)
    df_theta = np.nansum(vdf_dict['vdf'], axis=0)

    vx_plane_theta = vel_plane * np.cos(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vy_plane_theta = vel_plane * np.sin(np.radians(phi_plane)) * np.cos(np.radians(theta_plane))
    vz_plane_theta = vel_plane *                                 np.sin(np.radians(theta_plane))

    # converting the VDF as a function of energy and theta to log space
    log_df_theta_span = f.gen_log_df(df_theta).T

    # plotting the VDF before and after removing the non-contiguous pixels
    plot_preprocessing(np.log10(df_theta), log_df_theta_span.T,
                       vx_plane_theta, vz_plane_theta,
                       np.unique(vel_plane), np.unique(np.radians(theta_plane)), np.unique(np.radians(phi_plane)))

    plot_gap_identification(vx_plane_theta, vz_plane_theta, np.unique(vel_plane), log_df_theta_span)

    # plot_preprocessing_method(df_theta)