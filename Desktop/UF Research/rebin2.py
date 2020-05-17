import numpy as np
from copy import deepcopy
from astropy.table import Table

# Structure of incoming data will always consist of time stamps, flux values,
# and exposure times.

# There will be gaps in the data. These gaps can be replaced with either the
# median flux of the time series, or left at 0.

# TERMINOLOGY:
#
#    bins - the evenly-spaced "new" bins that the flux data will be distributed
#           into
#
#    binwidth - the width of the bins, in units of time
#
#    time - the time stamps from the original time series
#
#    flux - the flux data from the original time series
#
#    timeseries - the original time series (time and flux)
#
#    exptime - the exposure time for each frame of the time series
#
#    timestamp_position - the position of the time stamp relative to the
#                         exposure
#
#                   0   - the time stamp is at the beginning of each exposure
#                   0.5 - the time stamp is in the middle of each exposure
#                   1   - the time stamp is at the end of each exposure
#
#
#

def zipper(a, b):
    """
    Combines two arrays, fitting the values of b in between the values of a.
    """
    empty = np.empty((a.size + b.size), dtype=a.dtype)
    empty[0::2], empty[1::2] = a, b
    return empty


def rebin(timeseries, binwidth=None, exptime=None, timestamp_position=0.5,
          median_replace=True):
    """
    Rebin a time series into evenly-spaced bins, the number of which is a power
     of two for fast fourier transform compatibility.

    Parameters
    ----------
    timeseries : numpy array or astropy table
        The time series to be rebinned, consisting of two columns: timestamps
        and fluxes, in that order.
    binwidth : scalar, optional
        The width, in the same units of time as the timeseries, of each bin
        into which the fluxes from the timeseries will be distributed. If not
        provided, it will be inferred from the original time series.
    exptime : scalar, optional
        The exposure time, in the same units of time as the timeseries, of each
        image taken. If not provided, it will be inferred from the original
        time series.
    timestamp_position : scalar, optional
        The position of the time stamp relative to each exposure. Values are
        0 - the time stamp is at the beginning of each exposure; 0.5 - the time
        stamp is in the middle of each exposure; 1 - the time stamp is at the
        end of each exposure.
    median_replace : boolean, optional
        If true, gaps in the timeseries will be replaced with the median flux
        of all the data points.

    Returns
    -------
    a : array
        A two-column numpy array containing timestamps and fluxes in evenly-
        spaced bins. Timestamps indicate the beginning of each bin.
    """

    # FORMAT THE TIME SERIES

    if type(timeseries) != np.ndarray and type(timeseries) != Table:
        print('Error: timeseries data type is not a recognizable format.')
        return
    elif type(timeseries) == Table:
        time = np.array(timeseries.columns[0])
        flux = np.array(timeseries.columns[1])
        timeseries = np.vstack((time, flux)).T

    if np.shape(timeseries)[1] != 2:
        print('Error: input array is of the wrong dimension. Please input as '\
              'a two-column array with time and flux, in that order.')
        return

    # Unpack the timeseries
    time = timeseries[:,0]
    flux = timeseries[:,1]

    # If keyword arguments are unspecified, infer them here
    if exptime is None:
        dt = time[1:] - time[:-1]
        exptime = np.median(dt)

    if binwidth is None:
        dt = time[1:] - time[:-1]
        binwidth = np.median(dt)

    # Shift timestamps to be at beginning of exposures
    starttimes = time - timestamp_position*exptime
    endtimes = starttimes + exptime

    # Barycentered data may have overlapping time bins. Cut off the ends
    overlaps = endtimes[:-1] - starttimes[1:]
    endtimes[np.where(overlaps > 0)] = starttimes[np.where(overlaps > 0)[0]+1]

    # Find gaps in the timeseries
    gaps = starttimes[1:] - endtimes[:-1]
    gaps[np.abs(gaps)<1e-6]=0.0#

    # # Median replace, if set to true.
    median_flux = np.median(flux)
    # if median_replace is True:
    #     gaps = gaps*median_flux/exptime
    # else:
    #     gaps = gaps*0.

    # Inject gap replacements into original data
    gapflag1=np.zeros(len(flux))+1
    gapflag2=np.zeros(len(gaps))+1
    gapflag2[gaps>0]=0
    gapflag=zipper(gapflag1,gapflag2)
    print("gaps [:25]", gaps[:25])
    flux = zipper(flux, gaps)
    print("flux[:50]",flux[:50])
    gapidx=np.where(gapflag==0)
    for i in range(len(gapidx)):
        # flux[gapidx[i]]=((flux[gapidx[i]-1] + flux[gapidx[i]+1])*(flux[gapidx[i]]))/exptime
        # flux[gapidx[i]] = (flux[gapidx[i]]*median_flux)/exptime #global median replacement
        print("gapidx[i]-1",(flux[gapidx[i]-1]),"[gapidx[i]+1]",(flux[gapidx[i]+1]))#5/31
        flux[gapidx[i]]=(flux[gapidx[i]]*(flux[gapidx[i]-1]+flux[gapidx[i]+1]))/(2.*exptime)#5/31
    starttimes = zipper(starttimes, endtimes[:-1])
    endtimes = np.append(starttimes[1:], endtimes[-1])

    # Remove injected replacements where no gap was found
    indices_to_remove = np.where(starttimes == endtimes)
    flux = np.delete(flux, indices_to_remove)
    starttimes = np.delete(starttimes, indices_to_remove)
    endtimes = np.delete(endtimes, indices_to_remove)

    # Create the bins the flux will be redistributed into
    duration = np.max(time) + exptime - np.min(time)
    nbins = 2**np.ceil(np.log10(duration/binwidth)/np.log10(2)).astype(int)
    bins = np.zeros(nbins)

    # Bin start and stop times
    startbins = (starttimes[0] + binwidth*np.linspace(0, nbins-1, nbins))
    endbins = startbins + binwidth

    # Add final gap between last data point and end of the bins
    # flux = np.append(flux, (endbins[-1]-endtimes[-1])*median_flux/exptime)
    flux = np.append(flux, flux[-1])
    # print("flux[:25] after rebin",flux[:25]) #
    starttimes = np.append(starttimes, endtimes[-1])
    endtimes = np.append(endtimes, endbins[-1])

    # Split up timeseries flux into bins
    for i in range(len(flux)):

        j = np.where(startbins <= starttimes[i])[0][-1] # 1st bin left of start
        k = np.where(endbins >= endtimes[i])[0][0] # 1st bin right of end

        width = endtimes[i]-starttimes[i]

        if j == k:
            bins[j] += flux[i]

        if j < k:

            # Left bin
            frac_L = (startbins[j+1] - starttimes[i]) / width
            bins[j] += frac_L*flux[i]

            # Right bin
            frac_R = (endtimes[i] - endbins[k-1]) / width
            bins[k] += frac_R*flux[i]

        if j+1 < k:

            # Middle bins
            n = k - (j+1)
            frac = (endbins[k-1] - startbins[j+1]) / width / n
            bins[j+1:k] += frac*flux[i]

    print("bins[:25]",bins[:25])#
    return np.vstack((startbins, bins)).T
