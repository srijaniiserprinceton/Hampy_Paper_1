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

def plot_hammerhead(log_df_theta_post, vx, vz, cmap='jet', alpha=1.0):
    # making the subplot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    vmin, vmax = np.nanmin(log_df_theta_post), np.nanmax(log_df_theta_post)

    # plotting the post-processed df
    ax.contourf(vx, vz, log_df_theta_post, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlim([-650,None])
    ax.set_ylim([-400,600])
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(f'single_hammerhead/{hour}-{minute}-{second}-{cmap}.pdf')
    plt.close()

if __name__=='__main__':
    # timestamp for Verniero et al 2022 hammerhead
    year, month, date = 2020, 1, 29
    # hour, minute, second = 18, 10, 6
    hour, minute, second = 17, 55, 18
    # hour, minute, second = 18, 14, 24

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
    plot_hammerhead(log_df_theta_span.T, vx_plane_theta, vz_plane_theta)