import numpy as np
import datetime, pyspedas, glob, os, re
from sunpy.coordinates import spice,HeliographicCarrington
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
import astropy.units as u
import sunpy.coordinates
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.colors as mcolors
from matplotlib.dates import num2date

import encounters

# importing the spice kernel files
kernel_files = glob.glob("../spice_data/*.bsp")
spice.initialize(kernel_files)

def datetime2unix(dt_arr) :
    """Convert 1D array of `datetime.datetime` to unix timestamps"""
    return np.array([dt.timestamp() for dt in dt_arr])

def gen_dt_arr(dt_init, dt_final, cadence_days=1, incl_end=False):
    dt_mission = []

    if incl_end:
        while dt_init <= dt_final :
            dt_mission.append(dt_init)
            dt_init += datetime.timedelta(days=cadence_days)   
    else:
        while dt_init < dt_final :
            dt_mission.append(dt_init)
            dt_init += datetime.timedelta(days=cadence_days)

    dt_mission = np.array(dt_mission)

    return dt_mission


def gen_trajectory(dt_init, dt_final, cadence_days=1, spacecraft='SPP', incl_end=False):
    """
    'Generate Datetime Array'
    Get array of datetime.datetime from {dt_init} to {dt_final} every 
    {cadence_days} days
    """
    dt_mission = gen_dt_arr(dt_init, dt_final, cadence_days=cadence_days)

    # generating the trajectory
    parker_trajectory_inertial = spice.get_body(spacecraft, dt_mission)
    parker_trajectory_carrington = parker_trajectory_inertial.transform_to(HeliographicCarrington(observer="self"))
    parker_trajectory_carrington.representation_type = "spherical"

    return dt_mission, parker_trajectory_carrington

def interp_trajectory(dt_in, trajectory_in, dt_out, fixed_obstime=None) :
    trajectory_in.representation_type="spherical"
    radius_in = trajectory_in.radius.to("R_sun").value
    lon_in = trajectory_in.lon.to("deg").value
    lat_in = trajectory_in.lat.to("deg").value
    disconts = np.where(np.abs(np.diff(lon_in)) > 60)[0]
    nan_inds = []

    for discont in disconts: 
        nan_inds.append(discont)
        nan_inds.append(discont+1)

    radius_in[nan_inds] = np.nan
    lon_in[nan_inds] = np.nan
    lat_in[nan_inds] = np.nan

    if fixed_obstime is not None : obstime_out = fixed_obstime 
    else : obstime_out = dt_out

    return SkyCoord(radius=interp1d(datetime2unix(dt_in),radius_in,bounds_error=False)(datetime2unix(dt_out))*u.R_sun,
                    lon=interp1d(datetime2unix(dt_in),lon_in,bounds_error=False)(datetime2unix(dt_out))*u.deg,
                    lat=interp1d(datetime2unix(dt_in),lat_in,bounds_error=False)(datetime2unix(dt_out))*u.deg,        
                    representation_type="spherical",
                    frame=sunpy.coordinates.HeliographicCarrington,
                    obstime=obstime_out)

def get_ham_bins(tstart, tend, spice_cadence_days=1):
    # extracting the parker trajectory at a desired cadence
    dt_trajectory, parker_trajectory = gen_trajectory(tstart, tend, cadence_days=spice_cadence_days)

    # getting a 15min interpolation in the trajectory
    dt_trajectory_15min = gen_dt_arr(tstart, tend, cadence_days=1/(4*24))

    # download the L3 files
    tstart_str = f'{tstart.year:02d}-{tstart.month:02d}-{tstart.day:02d}/{tstart.hour:02d}:{tstart.minute:02d}:{tstart.second:02d}'
    tend_str = f'{tend.year:02d}-{tend.month:02d}-{tend.day:02d}/{tend.hour:02d}:{tend.minute:02d}:{tend.second:02d}'
    spi_vars = pyspedas.psp.spi(trange=[tstart_str, tend_str], datatype='spi_sf0a_l3_mom', level='l3', no_update=True,
                                time_clip=True, notplot=True, varnames=['psp_spi_DENS', 'psp_spi_MAGF_INST'])
    fld_vars = pyspedas.psp.fields(trange=[tstart_str, tend_str], datatype='mag_RTN_1min', level='l2',
                                   time_clip=True, notplot=True, no_update=True)
    dt_BR, BR = fld_vars['psp_fld_l2_mag_RTN_1min']['x'], fld_vars['psp_fld_l2_mag_RTN_1min']['y'][:,0]
    BR_15min = interp1d(datetime2unix(dt_BR), BR, bounds_error=False)(datetime2unix(dt_trajectory_15min))

    # extracting only the times we are interested in
    dt_span = spi_vars['psp_spi_DENS']['x']

    # obtaining the interpolated trajectory on the span timestamps
    span_cadence_trajectory = interp_trajectory(dt_trajectory, parker_trajectory, dt_span, fixed_obstime=None)
    fiftmin_cadence_trajectory = interp_trajectory(dt_trajectory, parker_trajectory, dt_trajectory_15min, fixed_obstime=None)

    return (dt_trajectory, parker_trajectory), (dt_span, span_cadence_trajectory), (dt_trajectory_15min, BR_15min, fiftmin_cadence_trajectory)

def get_days_around_perihelion(start_enc, end_enc, hamstring_dir, num_days_around_perihelion=10):
    enc_days = {}

    # finding which file indices in the hamstring folder falls within these dates
    filenames = os.listdir(hamstring_dir)

    # extracting the dates for each file
    filedates = np.asarray([np.datetime64(re.split('[_]', f)[1]).astype('datetime64[D]') for f in filenames])

    for enc in range(start_enc, end_enc+1):
        if(enc < 10): enc_str = f'E0{enc}'
        else: enc_str = f'E{enc}'

        pday = encounters.get_enc_dates(enc_str).astype('datetime64[D]')

        this_enc_days = np.arange(pday - np.timedelta64(num_days_around_perihelion, 'D'),
                                  pday + np.timedelta64(num_days_around_perihelion+1, 'D'))

        # keeping only the days which are in filedates (Hamstring dates)
        enc_days[enc_str] = this_enc_days[np.asarray([this_enc_day in filedates for this_enc_day in this_enc_days])]
        
    return enc_days


def dt64_to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.datetime.utcfromtimestamp(timestamp)

def angdist_on_sphere(coor1, coor2):
    '''
    Calculating the angular distance in degrees between two points assuming they are
    on a unit sphere.
    '''
    theta1, phi1 = coor1.lat.to(u.rad), coor1.lon.to(u.rad)
    theta2, phi2 = coor2.lat.to(u.rad), coor2.lon.to(u.rad)
    cos_theta = np.sin(theta1) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2) * np.cos(phi1 - phi2)

    # using small angle approximation in degrees
    return np.sqrt(2 * (1 - cos_theta)) * 180 / np.pi

import numpy as np
import astropy.units as u

def bin_by_angular_separation_fast(coords, max_sep_deg):
    """
    Group a 1D SkyCoord array into bins such that every point in a bin
    is within `max_sep_deg` of the *first* point in that bin.

    coords       : SkyCoord array (e.g. HeliographicCarrington, observer=None)
    max_sep_deg  : float, maximum angular separation in degrees

    Returns
    -------
    bins : list of SkyCoord
        Each element is coords[start:end] for one bin.
    """
    max_sep_deg = float(max_sep_deg)

    # Extract lon/lat ONCE as plain float arrays in radians
    lon = coords.lon.to_value(u.rad)   # shape (N,)
    lat = coords.lat.to_value(u.rad)   # shape (N,)

    valid = np.isfinite(lon) & np.isfinite(lat)
    n = lon.size
    bins = []
    bin_idx = np.zeros(len(lon), dtype='int')
    bin_count = 1

    i = 0
    while i < n:
        # Skip invalid points
        if not valid[i]:
            i += 1
            continue

        # Work within the next contiguous block of valid points
        # (stop at first invalid or end)
        j = i + 1
        while j < n and valid[j]:
            j += 1
        # Now the valid block is [i, j)

        start = i
        while start < j:
            ref_lon = lon[start]
            ref_lat = lat[start]

            # Vectorized angular distance from ref to ALL later points in this block
            lon_block = lon[start:j]
            lat_block = lat[start:j]

            dlon = lon_block - ref_lon
            dlat = lat_block - ref_lat

            # Great-circle distance (haversine formula) in radians
            sin_dlat2 = np.sin(dlat * 0.5)**2
            sin_dlon2 = np.sin(dlon * 0.5)**2
            a = sin_dlat2 + np.cos(ref_lat) * np.cos(lat_block) * sin_dlon2
            # Numerical safety
            a = np.clip(a, 0.0, 1.0)
            angle_rad = 2.0 * np.arcsin(np.sqrt(a))
            angle_deg = np.rad2deg(angle_rad)

            # Find first point that exceeds max_sep
            too_far_idx = np.where(angle_deg > max_sep_deg)[0]
            if too_far_idx.size == 0:
                # Entire remaining block belongs to this bin
                end = j
            else:
                end = start + too_far_idx[0]

            # Save this bin
            bins.append(coords[start:end])
            bin_idx[start:end] = bin_count
            bin_count += 1

            # Next bin starts where this one ended
            start = end

        # Move to first index after this valid block
        i = j

    return bins, bin_idx

def get_angular_bins_in_encounter(hammertimes, hamstring_dir, enc, angular_sep_deg = 1, spice_cadence_days=1, plot_trajectory=False):
    dt_start, dt_end = hammertimes[0], hammertimes[-1]

    spice_cadence, span_cadence, fifmin_cadence = get_ham_bins(dt_start, dt_end,
                                                               spice_cadence_days=spice_cadence_days)
    
    # calculate bins based on solid angle
    bins, bin_idx = bin_by_angular_separation_fast(span_cadence[1], angular_sep_deg)

    # counting the hammerheads and total VDF measurements per bin
    ham_counts_in_bin, all_counts_in_bin = count_points_in_bins(hammertimes, bins)

    # print([len(bins[i])//2 for i in range(len(bins))])
    bins_cen_lon = np.array([bins[i][len(bins[i])//2].lon.value for i in range(len(bins))])
    bins_cen_sinlat = np.sin(np.radians(np.array([bins[i][len(bins[i])//2].lat.value for i in range(len(bins))])))
    ham_frac_in_bin = ham_counts_in_bin / (1+all_counts_in_bin)

    # max size wille be 10
    size_frac = 200 / ham_frac_in_bin.max()

    if(plot_trajectory):
        fig, ax = plt.subplots(1, 2, figsize=(9,4), sharey=True)

        sin_carr_lat = np.sin(np.radians(fifmin_cadence[2].lat))
        carr_lon = fifmin_cadence[2].lon

        try:
            norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(fifmin_cadence[1]), vcenter=0, vmax=np.nanmax(fifmin_cadence[1]))
            ax[0].scatter(bins_cen_lon, bins_cen_sinlat, c='k', s=size_frac*ham_frac_in_bin, alpha=0.5, edgecolors='none')
            ax[0].scatter(carr_lon, sin_carr_lat, c=fifmin_cadence[1], s=2, cmap='bwr', norm=norm)
            ax[0].set_title(f'E{enc} | Good norm')
        except:
            ax[0].scatter(bins_cen_lon, bins_cen_sinlat, c='k', s=size_frac*ham_frac_in_bin, alpha=0.5, edgecolors='none')
            ax[0].scatter(carr_lon, sin_carr_lat, c=fifmin_cadence[1], s=2, cmap='bwr')
            ax[0].set_title(f'E{enc} | Bad norm')

        sin_carr_lat = np.sin(np.radians(span_cadence[1].lat))
        carr_lon = span_cadence[1].lon

        cmap = plt.cm.get_cmap('tab20')   # 20 discrete colors
        Ncolors = cmap.N 
        wrapped_colors = (bin_idx - 1) % Ncolors
        norm = plt.Normalize(vmin=0, vmax=Ncolors-1)

        ax[1].scatter(carr_lon[::10], sin_carr_lat[::10], c=wrapped_colors[::10], s=2, cmap=cmap, norm=norm)

        [ax[i].grid(True) for i in range(2)]
        ax[0].set_ylabel('Sine latitude')
        [ax[i].set_xlabel('Carrington Longitude [deg]') for i in range(2)]
        plt.tight_layout()

        plt.savefig(f'lat-lon-plots/{enc}.png')
        plt.close()
    
    return bins, bin_idx, ham_frac_in_bin, ham_counts_in_bin, all_counts_in_bin

def count_points_in_bins(hammertimes, bins):
    """
    Count how many `other_times` fall inside each time bin.
    
    All arrays must be sorted in increasing order.
    """
    bin_start_times = [bins[i].obstime[0].value for i in range(len(bins))]
    bin_end_times = [bins[i].obstime[-1].value for i in range(len(bins))]

    # index of first event >= start
    left_idx = np.searchsorted(hammertimes, bin_start_times, side='left')
    # index of first event > end
    right_idx = np.searchsorted(hammertimes, bin_end_times, side='right')

    # count = right - left
    ham_counts_in_bin = right_idx - left_idx

    # getting all the number of measurements for that bin
    all_counts_in_bin = np.array([len(bins[i]) for i in range(len(bins))])

    return ham_counts_in_bin, all_counts_in_bin

def lista_hista(times, tau, plot=False):
    """
    Generate a histogram from a list of datetime objects.
   
    Parameters:
        times : list of datetime.datetime
            Array of datetime values to bin.
        tau : float
            Temporal bin width in seconds.
        plot : bool or None
            If set, the histogram will be plotted.
   
    Returns:
        c : np.ndarray
            Histogram counts (NaNs if insufficient data).
        bin_centers_datetime : np.ndarray
            Bin centers as datetime objects (always returned).
    """
    if len(times) < 2:
        print("Not enough data to make a histogram. Returning NaNs.")
        # Still return a meaningful bin center array
        # We'll fake a single bin centered on the only point if it exists
        if len(times) == 1:
            single_center = times[0]
        else:
            single_center = None
        return np.full(1, np.nan), np.array([single_center], dtype='O')

    times_numeric = np.array([dt.timestamp() for dt in times])
    tlen = times_numeric[-1] - times_numeric[0]  # total length in seconds
    nbins = round(tlen / tau)
   
    if nbins < 1:
        print("Computed bins < 1 — returning NaNs.")
        return np.full(1, np.nan), np.array([times[0]], dtype='O')

    # Histogram the data
    c, b, _ = plt.hist(times, bins=nbins, histtype='step')

    # Calculate bin centers
    bin_centers = b[:-1] + np.diff(b) / 2
    bin_centers_datetime = np.array([num2date(center) for center in bin_centers])

    if plot:
        plt.plot(bin_centers_datetime, c)
        plt.xlabel('Time')
        plt.ylabel('Counts')
        plt.title('Histogram of Datetimes')
        plt.show()
   
    return c, bin_centers_datetime

if __name__=='__main__':
    # tstart, tend = datetime.datetime(2020, 1, 29), datetime.datetime(2020, 1, 29) #datetime.datetime(2025, 3, 29)

    start_enc, end_enc = 22, 23
    enc_days = get_days_around_perihelion(start_enc, end_enc, 'Hamstrings')

    for enc_key in enc_days.keys():
        spice_cadence, span_cadence, fifmin_cadence = get_ham_bins(dt64_to_datetime(enc_days[enc_key][0]),
                                                                   dt64_to_datetime(enc_days[enc_key][-1]),
                                                                   spice_cadence_days=1/(12*24))
        
        # calculate bins based on solid angle
        bins, bin_idx = bin_by_angular_separation_fast(span_cadence[1], 10)

        plt.figure(figsize=(7,4))

        sin_carr_lat = np.sin(np.radians(fifmin_cadence[2].lat))
        carr_lon = fifmin_cadence[2].lon

        try:
            norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(fifmin_cadence[1]), vcenter=0, vmax=np.nanmax(fifmin_cadence[1]))
            plt.scatter(carr_lon, sin_carr_lat, c=fifmin_cadence[1], s=2, cmap='bwr', norm=norm)
            plt.title('Good norm')
        except:
            plt.scatter(carr_lon, sin_carr_lat, c=fifmin_cadence[1], s=2, cmap='bwr')
            plt.title('Bad norm')
        plt.grid(True)
        plt.xlabel('Carrington Longitude [deg]')
        plt.xlabel('Sine latitude')
        plt.tight_layout()

        plt.savefig(f'lat-lon-plots/{enc_key}.pdf')
        plt.close()

        
