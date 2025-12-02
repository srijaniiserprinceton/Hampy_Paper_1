import glob, os, sys, datetime, getpass
from pathlib import Path
os.environ['SPEDAS_DATA_DIR']=os.path.join(Path.home(),"spedas")
import pyspedas
import numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from sunpy.coordinates import spice
from astropy.coordinates import SkyCoord
import astropy.units as u
import sunpy.coordinates
import astropy.constants as const
import pandas as pd
from sunpy.net import attrs as a, Fido
from sunpy.coordinates import sun
from sunkit_magex.pfss import utils as pfss_utils
def gen_dt_arr(dt_init,dt_final,cadence_days=1,incl_end=False) :
    """
    'Generate Datetime Array'
    Get array of datetime.datetime from {dt_init} to {dt_final} every 
    {cadence_days} days
    """
    dt_list = []
    if incl_end :
        while dt_init <= dt_final :
            dt_list.append(dt_init)
            dt_init += datetime.timedelta(days=cadence_days)   
    else :
        while dt_init < dt_final :
            dt_list.append(dt_init)
            dt_init += datetime.timedelta(days=cadence_days)
    return np.array(dt_list)
def datetime2unix(dt_arr) :
    """Convert 1D array of `datetime.datetime` to unix timestamps"""
    return np.array([dt.timestamp() for dt in dt_arr])
def unix2datetime(ut_arr) : 
    """Convert 1D array of unix timestamps (float) to `datetime.datetime`"""
    return np.array([datetime.datetime.utcfromtimestamp(ut) for ut in ut_arr])
def interp_trajectory(dt_in, trajectory_in, dt_out,fixed_obstime=None) :
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
    return SkyCoord(
        radius=interp1d(datetime2unix(dt_in),radius_in,bounds_error=False)(datetime2unix(dt_out))*u.R_sun,
        lon=interp1d(datetime2unix(dt_in),lon_in,bounds_error=False)(datetime2unix(dt_out))*u.deg,
        lat=interp1d(datetime2unix(dt_in),lat_in,bounds_error=False)(datetime2unix(dt_out))*u.deg,        
        representation_type="spherical",
        frame=sunpy.coordinates.HeliographicCarrington,
        obstime=obstime_out)
    
@u.quantity_input
def delta_long(r:u.R_sun,
               r_inner=2.5*u.R_sun,
               vsw=360.*u.km/u.s,
               omega_sun=14.713*u.deg/u.d,
               ):
    """ 
    Ballistic longitudinal shift of a Parker spiral connecting two
    points at radius r and r_inner, for a solar wind speed vsw. Solar
    rotation rate is also tunable
    """
    return (omega_sun * (r - r_inner) / vsw).to("deg")

def ballistically_project(skycoord,r_inner = 2.5*u.R_sun, vr_arr=None) :
    """
    Given a `SkyCoord` of a spacecraft trajectory in the Carrington frame,
    with `representation_type="spherical"`, and optionally an array of
    measured solar wind speeds at the same time intervals of the trajectory,
    return a SkyCoord for the trajectory ballistically projected down to 
    `r_inner` via a Parker spiral of the appropriate curvature. When `vr_arr`
    is not supplied, assumes wind speed is everywhere 360 km/s
    """
    if skycoord.representation_type != "spherical" :
        skycoord.representation_type="spherical"
    if vr_arr is None : vr_arr = np.ones(len(skycoord))*360*u.km/u.s
    lons_shifted = skycoord.lon + delta_long(skycoord.radius,
                                             r_inner=r_inner,
                                             vsw=vr_arr
                                            )
    return SkyCoord(
        lon = lons_shifted, 
        lat = skycoord.lat,
        radius = r_inner * np.ones(len(skycoord)),
        representation_type="spherical",
        frame = skycoord.frame
    )

def roll_to_180(lons) : 
    return ((lons + 180) % 360)-180
def d64_to_dt(d64_array) : return pd.to_datetime(d64_array).to_pydatetime()
def mean_filter(array,window=120) : return np.array(pd.Series(array).rolling(window,min_periods=10).mean().tolist())
def med_filter(array,window=120) : return np.array(pd.Series(array).rolling(window,min_periods=10).median().tolist())
from scipy.stats import mode
def mode_filter(array,window=120) : return np.array(pd.Series(array).rolling(window,min_periods=10).apply(lambda x: mode(x)[0]).tolist())
kernel_files = glob.glob("/home/sbadman/1_RESEARCH/DATA/spice_data/*.bsp")
spice.initialize(kernel_files)
spice.install_frame('IAU_SUN')
def parker_grid(ax,nrgrids = 5, nthgrids=13,rmin=10,rmax=100,annotate=True ) :
	rgrids = np.linspace(rmin,rmax,nrgrids)
	ax.set_xlim([-rmax,rmax])
	ax.set_ylim([-rmax,rmax])
	for R_ in rgrids  :
		ax.plot(R_*np.cos(np.linspace(0,2*np.pi,200)),
		R_*np.sin(np.linspace(0,2*np.pi,200)),
		color="black",linestyle="--",alpha=0.5,linewidth=0.8
		)
	if annotate : 
		for R_ in rgrids[:-1] : 
			ax.text(R_,0,f"{R_}"+"$R_\odot$",rotation=45,horizontalalignment="left")
	for phi in np.radians(np.linspace(0,360,nthgrids)) :
		ax.plot([0,rmax*np.cos(phi)],[0,rmax*np.sin(phi)],
		color="black",linestyle=":",alpha=0.5,linewidth=0.8
		)
	ax.set_aspect(1)
### Functions to Produce Parker Models
def Brp(R,R0=1*u.au,B0=3*u.nT) : 
    return (B0 * (R0/R)**2).to("nT")
def Btp(R,R0=1*u.au,B0=3*u.nT,vsw=360*u.km/u.s,Omega=14.73*u.deg/u.day) :
    return (-B0 * (R0/R)**2 * (Omega*R/vsw).to("rad").value).to("nT")
def Bnp(R) : return np.zeros(len(R))*u.nT
def Bp(R,R0=1*u.au,B0=3*u.nT,vsw=360*u.km/u.s,Omega=14.73*u.deg/u.day) :
    return (B0*(R0/R)**2 * (1 + (Omega*R/vsw).to("rad").value**2)**0.5).to("nT")
def phip(R,pol=-1,vsw=360*u.km/u.s,Omega=14.73*u.deg/u.day) : 
    return np.arctan2(-pol*(Omega*R/vsw).to("rad").value,pol) 
def thetap(R) : return np.zeros(len(R))
def Rot_3D(axis, angle) :
    """Return the rotation matrix specified by a rotation axis and an angle

    Following the Euler Axis-Angle parameterisation of rotations, this function  
    takes a given numpy 3x1 array (unit vector) specifying the axis to be
    rotated about, and an angle given in degrees, and returns the resulting
    rotation matrix. (Written by Samuel Badman and Chris Moeckel for the CURIE
    Cubesat project)
     
    Parameters
    ----------
    axis : 3x1 numpy array
        [n/a] Unit vector specifying axis of rotation. 
    angle : float
        [deg] Angle a vector would be rotated anticlockwise about the given axis

    Keywords
    --------
    
    Returns
    -------
    Rot: 3x3 numpy array
        [n/a] Rotation matrix corresponding to input axis and angle.


    Example
    -------
    >>> ax = numpy.array([ 0., 0., 1.])
    >>> print Rot_3D(ax, 90.0)
    [[ 0. -1.  0.]
     [ 1.  0.  0.]
     [ 0.  0.  1.]]

    Version
    -------
    07/11/17, SB, Initial Commit 
    01/01/25, SB, Make pure numpy
    """
    theta = angle             #convert to radians
    k  = axis/(np.dot(axis,axis))**0.5 #ensure normalised
    
    rot =  np.array([[np.cos(theta) + k[0]**2*(1.0-np.cos(theta)) , k[0]*k[1]* 
                      (1.0 - np.cos(theta)) - k[2]*np.sin(theta) ,  k[0]*k[2]*
                      (1.0 - np.cos(theta)) + k[1]*np.sin(theta)               ], 
                     [ k[0]*k[1]* (1.0 - np.cos(theta)) + k[2]*np.sin(theta) , 
                       np.cos(theta) + k[1]**2*(1.0 - np.cos(theta)) , 
                       k[1]*k[2]*(1.0 - np.cos(theta)) - k[0]*np.sin(theta)    ], 
                     [ k[0]*k[2]*(1.0 - np.cos(theta)) - k[1]*np.sin(theta) , 
                       k[1]*k[2]*(1.0 - np.cos(theta)) + k[0]*np.sin(theta) , 
                       np.cos(theta) + k[2]**2*(1.0 - np.cos(theta))           ]])
    
    rot[abs(rot) < 1e-16] = 0. #Get rid of floating point errors.
    
    return rot
    
def plot_periodic(x,y,xl=-180,xu=180,yl=-90,yu=90,figax=None,**kwargs) :

    ### Plot lines cleanly across periodic boundaries

    if figax is None : fig,ax=plt.gcf(),plt.gca()
    else : fig,ax=figax

    dtx = False
    if isinstance(x[0],u.Quantity) : x = np.array([x_.value for x_ in x])
    if isinstance(y[0],u.Quantity) : y = np.array([y_.value for y_ in y])

    xcrossings = np.where(np.abs(np.diff(x)) >= np.abs(xu-xl)/2)[0]
    xsplitx = np.split(x,1+xcrossings)
    ysplitx = np.split(y,1+xcrossings)

    plot_objs=[]
    for x_,y_ in zip(xsplitx,ysplitx) :
        ycrossings = np.where(np.abs(np.diff(y_)) >= np.abs(yu-yl)/2)[0]
        xsplitxy = np.split(x_,1+ycrossings)
        ysplitxy = np.split(y_,1+ycrossings)    
        for x__,y__ in zip(xsplitxy,ysplitxy) :
            if dtx : x__ = [datetime.utcfromtimestamp(tstamp) for tstamp in x__]
            plot_objs.append(ax.plot(x__,y__,**kwargs))
            if "label" in kwargs : kwargs.pop("label")
    return plot_objs
    
# function to generate one line -  parameters lon, radius of start, color
# Written by P. Planet, Summer 2024
def spiral_coords(lon:u.deg, start_radius:u.R_sun, vsw = 360 * u.km/u.s, end_radius = 220* u.R_sun, omega_sun=14.713*u.deg/u.d, radius_resolution = 200):
    rads = np.linspace(start_radius, end_radius, num = radius_resolution)
    lons = np.ones(np.size(rads)) * u.deg
    lons[0] = lon
    for i, rad in enumerate(rads):
        lons[i] = lon - delta_long(rad, r_inner=start_radius, vsw=vsw, omega_sun = omega_sun)
    return lons,rads

def create_euv_map(center_date,
                   euv_obs_cadence=1*u.day,
                   gaussian_filter_width = 30*u.deg,
                   days_around = 14, # number of days plus/minus the center date to create the map
                   save_dir=os.path.join('.','data'),
                   replace=False,
                   wvln = 193*u.angstrom,
                   dl_method = "jsoc",
                   shape_out = (720, 1440)
                   ) :
    '''
    Given `center_date`:`datetime.datetime` download a Carrington 
    rotation of EUV 193 Angstrom data centered around that date at
    an observational cadence of `euv_obs_cadence` (validate against
    some possible easy values). For each map, reproject into the 
    carrington frame of reference, and blend together with a 60 deg
    carrington longitude gaussian weighting to obtain a full sun EUV
    map. Save as fits file in `save_dir`:`str`
    '''
    assert dl_method in ["Fido","jsoc"]

    ## First, check if map centered on center_date has already been created 
    ## if it has, jump straight to the end.
    savepath = os.path.join(f"{save_dir}",f"{center_date.strftime('%Y-%m-%d')}_{int(wvln.value):04d}.fits")

    if not os.path.exists(savepath) or replace :

        ## Use sunpy Fido to search for AIA 193 data over a carrinton rotation
        ### If the downloads succeed once, they are archived on your computer in
        ### ~/sunpy/data and a database entry is created. This step then becomes 
        ### much quicker on the next run.
        sys.stdout.write(f"Searching for input EUV maps (via JSOC)")
        
        res = Fido.search(
            a.Time(center_date-datetime.timedelta(days=days_around), 
                center_date+datetime.timedelta(days=days_around)
            ),
            a.jsoc.Series('aia.lev1_euv_12s'),
            a.Wavelength(wvln),
            a.Sample(1*u.day),
            a.jsoc.Notify('samuel_badman@berkeley.edu')
        )

        ## Download, or return files if already exist
        downloaded_files = Fido.fetch(res)
            
        ## Read in downloaded data as `sunpy.map.Map` objects and downsample
        downsample_dims = [1024,1024] * u.pixel
        carrington_rotation = [sunpy.map.Map(m).resample(downsample_dims) 
                            for m in downloaded_files]
        
        ## Loop through input maps and reproject each one to the Carrington frame
        carrington_maps, datetime_list = [], []
        sys.stdout.write(f"Reprojecting {len(carrington_rotation)} Maps: \n")
        for ii,m in enumerate(carrington_rotation) :
            sys.stdout.write(f"{ii+1:02d}/{len(carrington_rotation)}\r")
            header =  sunpy.map.make_heliographic_header(
                m.date, m.observer_coordinate,shape_out, frame='carrington'
                )
            carrington_maps.append(m.reproject_to(header))
            datetime_list.append(datetime.datetime.strptime(m.meta['date-obs'], '%Y-%m-%dT%H:%M:%S.%f'))
        
        ## Combine maps together with gaussian weighting

        ### Make header for combined map (use central map as reference)  
        closest_datetime = min(datetime_list, key=lambda x: abs(x - center_date))
        ref_map = carrington_rotation[datetime_list.index(closest_datetime)]
        ref_coord = ref_map.observer_coordinate
        ref_header = sunpy.map.make_heliographic_header(
            ref_map.date, ref_coord, shape_out, frame="carrington"
        )

        ### Compute a gaussian weight for each pixel in each map.
        gaussian_weights = [
        np.exp(-((sunpy.map.all_coordinates_from_map(m).lon.to("deg").value 
                -sun.L0(m.date).to("deg").value + 180) % 360 - 180)**2
            /(2*gaussian_filter_width.to("deg").value**2)
            ) 
        for m in carrington_maps
        ]

        ### Average all maps together, rescale data to match 
        # maps from Badman+2022
        combined_map_data = np.nanmean([
                m.data*w for m,w in 
                zip(carrington_maps,gaussian_weights)
                ],
                axis=0)
        
        combined_map_gaussian_weights = sunpy.map.Map(combined_map_data, ref_header)
        

        ### Align LH edge with Carrington 0 for consistency
        combined_map_gaussian_weights_roll = pfss_utils.roll_map(
            combined_map_gaussian_weights)
        
        ## Save output combined map as fits file
        combined_map_gaussian_weights_roll.save(savepath,
                                                overwrite=replace)  

    ## Return output map filename
    return savepath
