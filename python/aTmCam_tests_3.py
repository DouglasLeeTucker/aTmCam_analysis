#!/usr/bin/env python

# Initial setup...
import numpy as np
import pandas as pd
from scipy import interpolate
import glob
import math
import matplotlib.pyplot as plt
from scipy.optimize import leastsq



#--------------------------------------------------------------------------
# Main code.
def main():
    

    # Let's grab all the observing logs for all three 3 years into a single sorted list...
    obsLogList = sorted(glob.glob('Test201?/???????-????????/???????-????????_obs_log.txt'))

    # Now load all the observing logs from the 3 years into a single pandas DataFrame...
    # (Sahar pointed me to this nifty trick from Stackoverflow:
    #  http://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe)
    obsdf_from_each_file = (pd.read_csv(obslog, usecols=['date-obs', 'mjd', '#expid', 'exptime1', 'exptime2', 'exptime3', 'exptime4'], delim_whitespace=True) for obslog in obsLogList)
    obsdf = pd.concat(obsdf_from_each_file, ignore_index=True)

    # NOTE:  WE LATER DISCOVERED THAT OBSDF CONTAINED LOTS OF DUPLICATE ROWS!
    #        THEREFORE, WE USE THE PANDAS DROP_DUPLICATES() FUNCTION TO REMOVE THESE HERE.

    obsdf.drop_duplicates(inplace=True)

    # We will also take the opportunity to sort by MJD here...
    obsdf.sort_values(by='mjd', inplace=True)

    # Let's add columns for an integer MJD and for UT [hours]...
    #  (Stole this idea from how we did this in SDSS!:)
    obsdf['imjd'] = np.floor(obsdf['mjd']+0.3).astype(int)
    obsdf['UT_hour'] = 24.*(obsdf['mjd'] - obsdf['imjd'])

    # And, as before, let's rename column "#expid" to "expid"...
    obsdf.rename(columns={'#expid': 'expid'}, inplace=True)

    # Oops... somehow, expid became a "float"...  let's re-cast it back to an "int"...
    obsdf.expid = obsdf.expid.astype(int)


    # Let's now grab all the instrumental magnitude files for all three 3 years into a single 
    #  sorted list...
    magLogList = sorted(glob.glob('Test201?/???????-????????/???????-????????_final_data_bin2.txt'))

    # Now load all the instrumental magnitude files from the 3 years into a single pandas DataFrame...
    # (Sahar pointed me to this nifty trick from Stackoverflow:
    #  http://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe)
    magdf_from_each_file = (pd.read_csv(maglog, delim_whitespace=True) for maglog in magLogList)
    magdf = pd.concat(magdf_from_each_file, ignore_index=True)

    # Let's add a column for an integer MJD...
    magdf['imjd'] = np.floor(magdf['mjd']+0.3).astype(int)

    # And, as before, let's rename column "#expid" to "expid"...
    magdf.rename(columns={'#expid': 'expid'}, inplace=True)

    # Again, we need to re-cast expid from a "float" back to an "int"...
    magdf.expid = magdf.expid.astype(int)

    # Now to merge the two mega-DataFrames...
    # Since the expids get reset each night, we need to match on both night *and* expid.
    #  (For "night", we will use the integer MJD, "imjd")...
    #  (We will use the suffix '_yyy' to indicate duplicate columns to drop later...)
    magobsdf = pd.merge(magdf, obsdf, on=['imjd','expid'], how='inner', suffixes=('', '_yyy'))

    # Create a list of duplicate columns to be removed...
    cols = [c for c in magobsdf.columns if c.lower()[-4:] == '_yyy']

    # Remove columns with "_yyy" as a suffix (i.e., the duplicate columns...)
    magobsdf.drop(cols, axis=1, inplace=True)
    
    # What is the frequency distribution of observations of the stars observed over all 3 years?
    magobsdf.object.value_counts()

    # Let's calculate the 3-year median mag for each star observed and subtract it off
    #  the mag for each observation of that star (and let's do this for each of mag1,
    #  mag2, mag3, and mag4)...

    objectname_list = magobsdf.object.unique()

    magobsdf.loc[:, 'median_mag1'] = -9999.
    magobsdf.loc[:, 'median_mag2'] = -9999.
    magobsdf.loc[:, 'median_mag3'] = -9999.
    magobsdf.loc[:, 'median_mag4'] = -9999.

    magobsdf.loc[:, 'dmag1'] = -9999.
    magobsdf.loc[:, 'dmag2'] = -9999.
    magobsdf.loc[:, 'dmag3'] = -9999.
    magobsdf.loc[:, 'dmag4'] = -9999.

    for objectname in objectname_list:
        print objectname,

        mask = (magobsdf.object==objectname)

        median_mag1 =  magobsdf[mask].mag1.median()
        median_mag2 =  magobsdf[mask].mag2.median()
        median_mag3 =  magobsdf[mask].mag3.median()
        median_mag4 =  magobsdf[mask].mag4.median()
    
        magobsdf.loc[mask, 'median_mag1'] = median_mag1
        magobsdf.loc[mask, 'median_mag2'] = median_mag1
        magobsdf.loc[mask, 'median_mag3'] = median_mag1
        magobsdf.loc[mask, 'median_mag4'] = median_mag1

        magobsdf.loc[mask, 'dmag1'] = magobsdf.loc[mask, 'mag1'] - median_mag1
        magobsdf.loc[mask, 'dmag2'] = magobsdf.loc[mask, 'mag2'] - median_mag2
        magobsdf.loc[mask, 'dmag3'] = magobsdf.loc[mask, 'mag3'] - median_mag3
        magobsdf.loc[mask, 'dmag4'] = magobsdf.loc[mask, 'mag4'] - median_mag4

        print median_mag1, median_mag2, median_mag3, median_mag4
    

    dmaglist = ['dmag1', 'dmag2', 'dmag3', 'dmag4']

    for dmag in dmaglist:

        print dmag
    
        # Let's plot dmag vs. MJD for 'HIP117542' for all 3 years...
        ax = magobsdf[(magobsdf.object=='HIP117452')].plot('mjd',dmag, grid=True, ylim=[-0.5,5.0])
        fig = ax.get_figure()
        outputFile = """./Temp/%s_vs_mjd_hip117542.png""" % (dmag)
        fig.savefig(outputFile)
        plt.close()
    
        # Let's plot dmag vs. MJD fo the 3 main Hipparcos in one plot...
        ax = magobsdf[((magobsdf.object=='HIP42334') | (magobsdf.object=='HIP75501') | (magobsdf.object=='HIP117452'))].plot('mjd',dmag, grid=True, ylim=[-0.5,5.0])
        fig = ax.get_figure()
        outputFile = """./Temp/%s_vs_mjd_hipbig3.png""" % (dmag)
        fig.savefig(outputFile)
        plt.close()

        # Let's plot a 2D histogram of log(Nobs), binned by dmag and airmass, for HIP117452...
        x=magobsdf[(magobsdf.object=='HIP117452')]['airmass']
        y=magobsdf[(magobsdf.object=='HIP117452')][dmag]
        xmin = 1.0
        xmax = 3.25
        ymin = -0.3
        ymax = 0.8
        fig, axs = plt.subplots(ncols=1)
        ax=axs
        hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='inferno')
        ax.axis([xmin, xmax, ymin, ymax])
        ax.set_title("HIP117452")
        ax.set_xlabel("airmass")
        ax.set_ylabel(dmag)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('log10(N)')
        plt.grid(True)
        ax.grid(color='white')
        #plt.show()
        outputFile = """./Temp/%s_vs_airmass_logN.HIP117452.png""" % (dmag)
        fig.savefig(outputFile)
        plt.close()

        # Let's plot a 2D histogram of log(Nobs), binned by dmag and airmass, for the Big 3...
        x=magobsdf[((magobsdf.object=='HIP42334') | (magobsdf.object=='HIP75501') | (magobsdf.object=='HIP117452'))]['airmass']
        y=magobsdf[((magobsdf.object=='HIP42334') | (magobsdf.object=='HIP75501') | (magobsdf.object=='HIP117452'))][dmag]
        xmin = 1.0
        xmax = 3.25
        ymin = -0.3
        ymax = 0.8
        fig, axs = plt.subplots(ncols=1)
        ax=axs
        hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='inferno')
        ax.axis([xmin, xmax, ymin, ymax])
        ax.set_title("HIP117452, HIP75501, and HIP42334 (The Big 3)")
        ax.set_xlabel("airmass")
        ax.set_ylabel(dmag)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('log10(N)')
        plt.grid(True)
        ax.grid(color='white')
        #plt.show()
        outputFile = """./Temp/%s_vs_airmass_logN.big3.png""" % (dmag)
        fig.savefig(outputFile)
        plt.close()

        # Let's plot a 2D histogram of log(Nobs), binned by "dmag-dmag3" and airmass, for HIP117452...
        # ... but skip if dmag is dmag3...
        if dmag=='dmag3': continue
        x=magobsdf[(magobsdf.object=='HIP117452')]['airmass']
        y1=magobsdf[(magobsdf.object=='HIP117452')][dmag]
        y2=magobsdf[(magobsdf.object=='HIP117452')]['dmag3']
        y=y1-y2
        xmin = 1.0
        xmax = 3.25
        ymin = -0.3
        ymax = 0.8
        fig, axs = plt.subplots(ncols=1)
        ax=axs
        hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='inferno')
        ax.axis([xmin, xmax, ymin, ymax])
        ax.set_title("HIP117452")
        ax.set_xlabel("airmass")
        ylabel="""%s-dmag3""" % (dmag)
        ax.set_ylabel(ylabel)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('log10(N)')
        plt.grid(True)
        ax.grid(color='white')
        #plt.show()
        outputFile = """./Temp/%s3_vs_airmass_logN.HIP117452.png""" % (dmag)
        fig.savefig(outputFile)
        plt.close()

        # Let's plot a 2D histogram of log(Nobs), binned by "dmag-dmag3" and airmass, for the Big 3...
        # ... but skip if dmag is dmag3...
        if dmag=='dmag3': continue
        x=magobsdf[((magobsdf.object=='HIP42334') | (magobsdf.object=='HIP75501') | (magobsdf.object=='HIP117452'))]['airmass']
        y1=magobsdf[((magobsdf.object=='HIP42334') | (magobsdf.object=='HIP75501') | (magobsdf.object=='HIP117452'))][dmag]
        y2=magobsdf[((magobsdf.object=='HIP42334') | (magobsdf.object=='HIP75501') | (magobsdf.object=='HIP117452'))]['dmag3']
        y=y1-y2
        xmin = 1.0
        xmax = 3.25
        ymin = -0.3
        ymax = 0.8
        fig, axs = plt.subplots(ncols=1)
        ax=axs
        hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='inferno')
        ax.axis([xmin, xmax, ymin, ymax])
        ax.set_title("HIP117452, HIP75501, and HIP42334 (The Big 3)")
        ax.set_xlabel("airmass")
        ylabel="""%s-dmag3""" % (dmag)
        ax.set_ylabel(ylabel)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('log10(N)')
        plt.grid(True)
        ax.grid(color='white')
        #plt.show()
        outputFile = """./Temp/%s3_vs_airmass_logN.big3.png""" % (dmag)
        fig.savefig(outputFile)
        plt.close()


    # Precipitable Water Vapor (PWV)

    # The Suominet-processed GPSmon data are in one big CSV file, updated daily...
    gpsdatafile = 'SuominetResults.no_whitespace.csv'

    # Read it into a pandas DataFrame
    gpsdf = pd.read_csv(gpsdatafile)

    # Let's make sure the DataFrame is sorted by MJD...
    gpsdf.sort_values('MJD',inplace=True)

    # Grab a clean sample...
    mask = ( (gpsdf.PWV >= 0.00) & (gpsdf.PWV_ERR < 2.0) & (gpsdf.Pressure > 700.) & (gpsdf.Pressure < 850.) & (gpsdf.Temp > -20.) & (gpsdf.Temp < 50.) )

    clean_gpsdf = gpsdf[mask].copy()

    # Plot the GPSmon PWV vs. MJD...
    ax = clean_gpsdf.plot('MJD','PWV', grid=True, ylim=[-5,25.0])
    fig = ax.get_figure()
    outputFile = "./Temp/gps_pwv_vs_mjd.png"
    fig.savefig(outputFile)
    plt.close()

    # For future reference, let's add an integer MJD column to the GPSmon DataFrame...
    clean_gpsdf['imjd'] = np.floor(clean_gpsdf['MJD']+0.3).astype(int)

    # Create a sorted list of all imjds that lie within both the GPS and the aTmCam data sets...
    mjd_atm_gps_list = sorted(list(set(clean_gpsdf['imjd'].unique()) & set(magobsdf['imjd'].unique())))

    # Create a linear interpolation function for the GPS PWV data...
    gps_pwv_interp1d = interpolate.interp1d(clean_gpsdf['MJD'], clean_gpsdf['PWV'], kind='linear')

    # Grab the list of MJDs from the aTmCam instrumental mag DataFrame...
    mjd_array = np.array(magobsdf['mjd'])

    # Find the (interpolated) GPSmon PWV values for the MJDs from the aTmCam instr. mag DataFrame...
    pwv_array = gps_pwv_interp1d(mjd_array)

    # Convert the mjd_array and pwv_array into their own Pandas DataFrame...
    mjd_series = pd.Series(mjd_array, name='mjd')
    pwv_series = pd.Series(pwv_array, name='pwv')
    mjdpwvdf = pd.concat([mjd_series, pwv_series], axis=1)

    # Match (merge) the GPSmon PWV values with the aTmCam instr. mag DataFrame...
    magobsdf_new = pd.merge(magobsdf, mjdpwvdf, on='mjd', how='inner')


    # Let's plot a 2D histogram of log(Nobs), binned by delta_mag4 and PWV, for 'HIP117452'...
    x=magobsdf_new[(magobsdf_new.object=='HIP117452')].pwv
    y=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag4
    xmin = 0.0
    xmax = 12.0
    ymin = -0.3
    ymax = 0.8
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='inferno')
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title("HIP117452")
    ax.set_xlabel("PWV")
    ax.set_ylabel("dmag4")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(N)')
    plt.grid(True)
    ax.grid(color='white')
    #plt.show()
    outputFile = "./Temp/dmag4_vs_pwv_logN.HIP117452.png"
    fig.savefig(outputFile)
    plt.close()


    # If we plot dmag4-dmag3, we can remove the effects of clouds...
    x=magobsdf_new[(magobsdf_new.object=='HIP117452')].pwv
    y1=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag4
    y2=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag3
    y=y1-y2
    xmin = 0.0
    xmax = 12.0
    ymin = -0.3
    ymax = 0.8
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='inferno')
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title("HIP117452")
    ax.set_xlabel("PWV")
    ax.set_ylabel("dmag4-dmag3")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(N)')
    plt.grid(True)
    ax.grid(color='white')
    #plt.show()
    outputFile = "./Temp/dmag43_vs_pwv_logN.HIP117452.png"
    fig.savefig(outputFile)
    plt.close()

    # This is like the original delta_mag4 vs. airmass plot, but now color-coded by PWV...
    x=magobsdf_new[(magobsdf_new.object=='HIP117452')].airmass
    y=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag4
    z=magobsdf_new[(magobsdf_new.object=='HIP117452')].pwv
    xmin = 1.0
    xmax = 3.25
    ymin = -0.3
    ymax = 0.8
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, C=z, gridsize=100, cmap='rainbow', reduce_C_function=np.median)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title("HIP117452")
    ax.set_xlabel("airmass")
    ax.set_ylabel("dmag4")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('PWV')
    plt.grid(True)
    ax.grid(color='white')
    #plt.show()
    outputFile = "./Temp/dmag4_vs_airmass_pwv.HIP117452.png"
    fig.savefig(outputFile)
    plt.close()

    # Same as above, but now dmag4-dmag3 vs. airmass, color-coded by PWV...
    x=magobsdf_new[(magobsdf_new.object=='HIP117452')].airmass
    y1=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag4
    y2=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag3
    y=y1-y2
    z=magobsdf_new[(magobsdf_new.object=='HIP117452')].pwv
    xmin = 1.0
    xmax = 3.25
    ymin = -0.8
    ymax = 0.8
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, C=z, gridsize=100, cmap='rainbow', reduce_C_function=np.median)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title("HIP117452")
    ax.set_xlabel("airmass")
    ax.set_ylabel("dmag4-dmag3")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('PWV')
    plt.grid(True)
    ax.grid(color='white')
    #plt.show()
    outputFile = "./Temp/dmag43_vs_airmass_pwv.HIP117452.png"
    fig.savefig(outputFile)
    plt.close()

    # Let's switch things around a bit, plotting dmag4 vs. PWV and color-coding by airmass...
    z=magobsdf_new[(magobsdf_new.object=='HIP117452')].airmass
    y=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag4
    x=magobsdf_new[(magobsdf_new.object=='HIP117452')].pwv
    xmin = 0.0
    xmax = 12.0
    ymin = -0.3
    ymax = 0.8
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, C=z, gridsize=100, cmap='rainbow', reduce_C_function=np.median)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title("HIP117452")
    ax.set_xlabel("PWV")
    ax.set_ylabel("dmag4")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Airmass')
    plt.grid(True)
    ax.grid(color='white')
    #plt.show()
    outputFile = "./Temp/dmag4_vs_pwv_airmass.HIP117452.png"
    fig.savefig(outputFile)
    plt.close()

    # Same as two figures above, but switching PWV and airmass axes...
    z=magobsdf_new[(magobsdf_new.object=='HIP117452')].airmass
    y1=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag4
    y2=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag3
    y=y1-y2
    x=magobsdf_new[(magobsdf_new.object=='HIP117452')].pwv
    xmin = 0.0
    xmax = 12.0
    ymin = -0.8
    ymax = 0.8
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, C=z, gridsize=100, cmap='rainbow', reduce_C_function=np.median)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title("HIP117452")
    ax.set_xlabel("PWV")
    ax.set_ylabel("dmag4-dmag3")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Airmass')
    plt.grid(True)
    ax.grid(color='white')
    #plt.show()
    outputFile = "./Temp/dmag43_vs_pwv_airmass.HIP117452.png"
    fig.savefig(outputFile)
    plt.close()

    #magobsdf_new.to_csv('/Users/dtucker/Desktop/magobsdf_new.csv', index=False)


    #
    # Let's work a bit with just the HIP117452 data from before MJD 57100...
    #

    # We will call the new dataframe df...
    df_orig = magobsdf_new[( (magobsdf_new.mjd < 57000.) & (magobsdf_new.object=='HIP117452') )].copy()
    df = df_orig.copy()

    # Since the mag3 filter has almost no airmass dependence, we will use it to do an initial
    # identification and masking of cloudy exposures...
    
    # First, create a mask that removes the most egregiously bad mag3 exposures...
    mask1 = ( np.abs(df['dmag3']) < 1.0 )

    # Then, identify and and mask the worst 5% of mag3 exposures remaining...
    df['vigintile'] = pd.qcut( df['dmag3'], 20, labels=False )
    mask2 = (df['vigintile']<19)

    # Combine mask1 and mask2
    mask = mask1 & mask2

    # Do an iterative loop to remove the additional, noticeably cloudy exposures...
    niter = 9
    nsigma = 3.0
    for i in range(niter):

        iiter = i + 1
        print """   iter%d...""" % ( iiter )

        # make a copy of original df, and then delete the old one...
        df = df[mask].copy()
        #del df

        # Fit a straight line to dmag3 vs. mjd...
        z = np.polyfit(x=df.loc[:, 'mjd'], y=df.loc[:, 'dmag3'], deg=1)
        p = np.poly1d(z)

        mag_per_100_days = 100*z[0]
    
        # Confirm this:
        #df.loc[:,'res'] = df.dmag3 - p(df.mjd)
        df.loc[:,'res'] = df.loc[:,'dmag3'] - p(df.mjd)
        #df.loc[:,'res'] = df.loc[:,'dmag3'] - p(df.loc[:,'mjd'])

        stddev = df['res'].std()
        mask = (np.abs(df.res)< nsigma*stddev)
    
        nrows = df['dmag3'].size
        print """  RMS:  %f\nNumber of rows remaining:  %d""" % ( stddev, nrows )
        print """  mags per 100 days:  %f""" % (mag_per_100_days)
        print 
    
        #df = newdf

        ax=df.plot('mjd','dmag3', grid=True, kind='scatter')
        plt.plot(df.mjd,p(df.mjd),'m-')
        fig = ax.get_figure()
        outputFile = """./Temp/dmag3_vs_mjd_hip117542_mjdfit_iter%d.png""" % (iiter)
        fig.savefig(outputFile)
        plt.close()

        ax=df.hist('res', grid=True, bins=100)
        ax=df['res'].hist(grid=True, bins=100)
        fig = ax.get_figure()
        outputFile = """./Temp/dmag3_vs_mjd_hip117542_mjdfit_reshist_iter%d.png""" % (iiter)
        fig.savefig(outputFile)
        plt.close()

    
    #endloop
    
    df.drop('vigintile', axis=1, inplace=True)

    # Add a dmjd column to df...
    df.loc[:,'dmjd'] = df.loc[:,'mjd'] - df.mjd.min()

    # Save a cleaned copy of the original data....
    df_orig_clean = df.copy()


    # Now that the obviously cloudy/bad exposures have been removed, let's do a fit of dmag vs. (dmjd,airmass)

    for dmag in dmaglist:

        # We need to do something special for dmag4, so skip it for now...
        if dmag=='dmag4': continue

        print dmag

        df = df_orig_clean.copy()
        
        # Create initial (and generous)  mask...
        mask = ( df['airmass'] < 10.0 )

        # Sigma-clipping parameters...
        nsigma = 3.0
        niter = 3

        for i in range(niter):

            iiter = i + 1
            print """   iter%d...""" % ( iiter )

            # make a copy of original df, and then delete the old one...
            df = df[mask].copy()
            #del df

            p,rms = aTmCamTestFit(df.loc[:,'dmjd'], df.loc[:,'airmass'], df.loc[:,dmag])
            df.loc[:,'res'] = residuals(p,df.loc[:,'dmjd'],df.loc[:,'airmass'],df.loc[:,dmag])

            stddev = df['res'].std()
            mask = (np.abs(df.res)< nsigma*stddev)
    
            ax=df.plot('mjd','res', grid=True, kind='scatter')
            #plt.plot(df.mjd,p(df.mjd),'m-')
            fig = ax.get_figure()
            outputFile = """./Temp/%sres_vs_mjd_hip117542_mjdfit_iter%d.png""" % (dmag, iiter)
            fig.savefig(outputFile)
            plt.close()

            ax=df.plot('airmass','res', grid=True, kind='scatter')
            #plt.plot(df.mjd,p(df.mjd),'m-')
            fig = ax.get_figure()
            outputFile = """./Temp/%sres_vs_airmass_hip117542_mjdfit_iter%d.png""" % (dmag, iiter)
            fig.savefig(outputFile)
            plt.close()

            ax=df['res'].hist(grid=True, bins=100)
            fig = ax.get_figure()
            outputFile = """./Temp/%sres_vs_mjd_hip117542_mjdfit_reshist_iter%d.png""" % (dmag, iiter)
            fig.savefig(outputFile)
            plt.close()

    
        # Let's plot a 2D histogram of log(Nobs), binned by dmagres and airmass, for HIP117452...
        x=df['airmass']
        y=df['res']
        xmin = 1.0
        xmax = 3.25
        ymin = -3.0*df.res.std()
        ymax =  3.0*df.res.std()
        #ymin = -0.1
        #ymax = 0.1
        #ymin = -1.0*df.res.std()
        #ymax = df.res.std()
        fig, axs = plt.subplots(ncols=1)
        ax=axs
        hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='inferno')
        ax.axis([xmin, xmax, ymin, ymax])
        ax.set_title("HIP117452")
        ax.set_xlabel("airmass")
        ylabel="""%sres"""  % (dmag)
        ax.set_ylabel(ylabel)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('log10(N)')
        plt.grid(True)
        ax.grid(color='white')
        #plt.show()
        outputFile = """./Temp/%sres_vs_airmass_logN.HIP117452.png""" % (dmag)
        fig.savefig(outputFile)
        plt.close()
    
    
        # Let's plot a 2D histogram of PWV, binned by dmagres and airmass, for HIP117452...
        x=df['airmass']
        y=df['res']
        z=df['pwv']
        xmin = 1.0
        xmax = 3.25
        ymin = -3.0*df.res.std()
        ymax =  3.0*df.res.std()
        #ymin = -0.1
        #ymax = 0.1
        fig, axs = plt.subplots(ncols=1)
        ax=axs
        hb = ax.hexbin(x, y, C=z, gridsize=100, cmap='rainbow', reduce_C_function=np.median)
        ax.axis([xmin, xmax, ymin, ymax])
        ax.set_title("HIP117452")
        ax.set_xlabel("airmass")
        ylabel="""%sres"""  % (dmag)
        ax.set_ylabel(ylabel)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('PWV')
        plt.grid(True)
        ax.grid(color='white')
        #plt.show()
        outputFile = """./Temp/%sres_vs_airmass_pwv.HIP117452.png"""% (dmag)
        fig.savefig(outputFile)
        plt.close()


        # Let's plot a 2D histogram of log(Nobs), binned by dmagres and mjd, for HIP117452...
        x=df['mjd']
        y=df['res']
        xmin = df.mjd.min()
        xmax = df.mjd.max()
        ymin = -3.0*df.res.std()
        ymax =  3.0*df.res.std()
        #ymin = -0.1
        #ymax = 0.1
        fig, axs = plt.subplots(ncols=1)
        ax=axs
        hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='inferno')
        ax.axis([xmin, xmax, ymin, ymax])
        ax.set_title("HIP117452")
        ax.set_xlabel("MJD")
        ylabel="""%sres"""  % (dmag)
        ax.set_ylabel(ylabel)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('log10(N)')
        plt.grid(True)
        ax.grid(color='white')
        #plt.show()
        outputFile = """./Temp/%sres_vs_mjd_logN.HIP117452.png"""% (dmag)
        fig.savefig(outputFile)
        plt.close()
    
    
        # Let's plot a 2D histogram of PWV, binned by dmagres and mjd, for HIP117452...
        x=df['mjd']
        y=df['res']
        z=df['pwv']
        xmin = df.mjd.min()
        xmax = df.mjd.max()
        ymin = -3.0*df.res.std()
        ymax =  3.0*df.res.std()
        #ymin = -0.1
        #ymax = 0.1
        fig, axs = plt.subplots(ncols=1)
        ax=axs
        hb = ax.hexbin(x, y, C=z, gridsize=100, cmap='rainbow', reduce_C_function=np.median)
        ax.axis([xmin, xmax, ymin, ymax])
        ax.set_title("HIP117452")
        ax.set_xlabel("MJD")
        ylabel="""%sres"""  % (dmag)
        ax.set_ylabel(ylabel)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('PWV')
        plt.grid(True)
        ax.grid(color='white')
        #plt.show()
        outputFile = """./Temp/%sres_vs_mjd_pwv.HIP117452.png"""% (dmag)
        fig.savefig(outputFile)
        plt.close()


        
    # Now we work on dmag4...
    dmag = 'dmag4'
    print dmag

    df = df_orig_clean.copy()
        
    # Create initial (and generous)  mask, ESPECIALLY FOR mag4...
    mask = ( ( df['airmass'] < 10.0 ) & ( df['pwv'] < 8.0 ) )

    # Sigma-clipping parameters...
    nsigma = 3.0
    niter = 3

    for i in range(niter):

        iiter = i + 1
        print """   iter%d...""" % ( iiter )

        # make a copy of original df, and then delete the old one...
        df = df[mask].copy()
        #del df

        p,rms = aTmCamTestFit4(df.loc[:,'dmjd'], df.loc[:,'airmass'], df.loc[:,'pwv'], df.loc[:,dmag])
        df.loc[:,'res'] = residuals4(p,df.loc[:,'dmjd'],df.loc[:,'airmass'],df.loc[:,'pwv'],df.loc[:,dmag])

        stddev = df['res'].std()
        mask = (np.abs(df.res)< nsigma*stddev)
    
        ax=df.plot('mjd','res', grid=True, kind='scatter')
        #plt.plot(df.mjd,p(df.mjd),'m-')
        fig = ax.get_figure()
        outputFile = """./Temp/%sres_vs_mjd_hip117542_mjdfit_iter%d.png""" % (dmag, iiter)
        fig.savefig(outputFile)
        plt.close()

        ax=df.plot('airmass','res', grid=True, kind='scatter')
        #plt.plot(df.mjd,p(df.mjd),'m-')
        fig = ax.get_figure()
        outputFile = """./Temp/%sres_vs_airmass_hip117542_mjdfit_iter%d.png""" % (dmag, iiter)
        fig.savefig(outputFile)
        plt.close()

        ax=df.plot('pwv','res', grid=True, kind='scatter')
        #plt.plot(df.mjd,p(df.mjd),'m-')
        fig = ax.get_figure()
        outputFile = """./Temp/%sres_vs_pwv_hip117542_mjdfit_iter%d.png""" % (dmag, iiter)
        fig.savefig(outputFile)
        plt.close()

        ax=df['res'].hist(grid=True, bins=100)
        fig = ax.get_figure()
        outputFile = """./Temp/%sres_vs_mjd_hip117542_mjdfit_reshist_iter%d.png""" % (dmag, iiter)
        fig.savefig(outputFile)
        plt.close()

    
    # Let's plot a 2D histogram of log(Nobs), binned by dmagres and airmass, for HIP117452...
    x=df['airmass']
    y=df['res']
    xmin = 1.0
    xmax = 3.25
    ymin = -3.0*df.res.std()
    ymax =  3.0*df.res.std()
    #ymin = -0.1
    #ymax = 0.1
    #ymin = -1.0*df.res.std()
    #ymax = df.res.std()
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='inferno')
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title("HIP117452")
    ax.set_xlabel("airmass")
    ylabel="""%sres"""  % (dmag)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(N)')
    plt.grid(True)
    ax.grid(color='white')
    #plt.show()
    outputFile = """./Temp/%sres_vs_airmass_logN.HIP117452.png""" % (dmag)
    fig.savefig(outputFile)
    plt.close()
    
    
    # Let's plot a 2D histogram of PWV, binned by dmagres and airmass, for HIP117452...
    x=df['airmass']
    y=df['res']
    z=df['pwv']
    xmin = 1.0
    xmax = 3.25
    ymin = -3.0*df.res.std()
    ymax =  3.0*df.res.std()
    #ymin = -0.1
    #ymax = 0.1
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, C=z, gridsize=100, cmap='rainbow', reduce_C_function=np.median)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title("HIP117452")
    ax.set_xlabel("airmass")
    ylabel="""%sres"""  % (dmag)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('PWV')
    plt.grid(True)
    ax.grid(color='white')
    #plt.show()
    outputFile = """./Temp/%sres_vs_airmass_pwv.HIP117452.png"""% (dmag)
    fig.savefig(outputFile)
    plt.close()


    # Let's plot a 2D histogram of log(Nobs), binned by dmagres and mjd, for HIP117452...
    x=df['mjd']
    y=df['res']
    xmin = df.mjd.min()
    xmax = df.mjd.max()
    ymin = -3.0*df.res.std()
    ymax =  3.0*df.res.std()
    #ymin = -0.1
    #ymax = 0.1
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='inferno')
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title("HIP117452")
    ax.set_xlabel("MJD")
    ylabel="""%sres"""  % (dmag)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(N)')
    plt.grid(True)
    ax.grid(color='white')
    #plt.show()
    outputFile = """./Temp/%sres_vs_mjd_logN.HIP117452.png"""% (dmag)
    fig.savefig(outputFile)
    plt.close()
    
    
    # Let's plot a 2D histogram of PWV, binned by dmagres and mjd, for HIP117452...
    x=df['mjd']
    y=df['res']
    z=df['pwv']
    xmin = df.mjd.min()
    xmax = df.mjd.max()
    ymin = -3.0*df.res.std()
    ymax =  3.0*df.res.std()
    #ymin = -0.1
    #ymax = 0.1
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, C=z, gridsize=100, cmap='rainbow', reduce_C_function=np.median)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title("HIP117452")
    ax.set_xlabel("MJD")
    ylabel="""%sres"""  % (dmag)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('PWV')
    plt.grid(True)
    ax.grid(color='white')
    #plt.show()
    outputFile = """./Temp/%sres_vs_mjd_pwv.HIP117452.png"""% (dmag)
    fig.savefig(outputFile)
    plt.close()


    # Let's plot a 2D histogram of log(Nobs), binned by dmagres and pwv, for HIP117452...
    x=df['pwv']
    y=df['res']
    xmin = df.pwv.min()
    xmax = df.pwv.max()
    ymin = -3.0*df.res.std()
    ymax =  3.0*df.res.std()
    #ymin = -0.1
    #ymax = 0.1
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='inferno')
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title("HIP117452")
    ax.set_xlabel("PWV")
    ylabel="""%sres"""  % (dmag)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(N)')
    plt.grid(True)
    ax.grid(color='white')
    #plt.show()
    outputFile = """./Temp/%sres_vs_pwv_logN.HIP117452.png"""% (dmag)
    fig.savefig(outputFile)
    plt.close()
    
    

    ## Extract the series...
    #dmjd_series = df['mjd'] - df.mjd.min()
    #airmass_series = df['airmass']
    #dmag_series = df['dmag3']
    #
    ## Convert to numpy arrays.
    #dmjd_array = dmjd_series.values
    #airmass_array = airmass_series.values
    #dmag_array = dmag_series.values
    #
    #p,rms = aTmCamTestFit(dmjd_array, airmass_array, dmag_array)
    #
    ## Recalculate the mask to sigma-clip the residuals...
    #res=residuals(p,dmjd_array,airmass_array,dmag_array)
    #mask = (abs(res) < nsigma*rms)
    #dmjd_array = dmjd_array[np.where(mask)]
    #airmass_array = airmass_array[np.where(mask)]
    #dmag_array = dmag_array[np.where(mask)]

        
    print """That's all, folks!"""

    return 0
    


#--------------------------------------------------------------------------
# Parametric function:  
#  p is the parameter vector; 
def fp(p,dmjd_array,airmass_array):
    return p[0] + p[1]*dmjd_array + p[2]*airmass_array

#--------------------------------------------------------------------------
# Error function:
def residuals(p,dmjd_array,airmass_array,dmag_array):
    err = (dmag_array-fp(p,dmjd_array,airmass_array))
    return err

#--------------------------------------------------------------------------
# Fitting code:
def aTmCamTestFit(dmjd_array, airmass_array, dmag_array):

    # Calculate the median of dmag for use as an initial guess
    # for the overall zeropoint offset..
    mdn = np.median( dmag_array, None )

    # Parameter names
    pname = (['a_0', 'a_1', 'k'])

    # Initial parameter values
    p0 = [mdn, 0.0, 0.0]

    print 
    print 'Initial parameter values:  ', p0

    #print fp(p0,dmjd_array,airmass_array)
    #print residuals(p0,dmjd_array,airmass_array,dmag_array)

    # Perform fit

    p,cov,infodict,mesg,ier = leastsq(residuals, p0, args=(dmjd_array, airmass_array, dmag_array), maxfev=10000, full_output=1)

    if ( ier>=1 and ier <=4):
        print "Converged"
    else:
        print "Not converged"
        print mesg


    # Calculate some descriptors of the fit 
    # (similar to the output from gnuplot 2d fits)

    chisq=sum(infodict['fvec']*infodict['fvec'])
    dof=len(dmag_array)-len(p)
    rms=math.sqrt(chisq/dof)
    
    print "Converged with chi squared ",chisq
    print "degrees of freedom, dof ", dof
    print "RMS of residuals (i.e. sqrt(chisq/dof)) ", rms
    print "Reduced chisq (i.e. variance of residuals) ", chisq/dof
    print


    # uncertainties are calculated as per gnuplot, "fixing" the result
    # for non unit values of the reduced chisq.
    # values at min match gnuplot
    print "Fitted parameters at minimum, with 68% C.I.:"
    for i,pmin in enumerate(p):
        print "%-10s %13g +/- %13g   (%5f percent)" % (pname[i],pmin,math.sqrt(cov[i,i])*math.sqrt(chisq/dof),100.*math.sqrt(cov[i,i])*math.sqrt(chisq/dof)/abs(pmin))
    print


    print "Correlation matrix:"
    # correlation matrix close to gnuplot
    print "               ",
    for i in range(len(pname)): print "%-10s" % (pname[i],),
    print
    for i in range(len(p)):
        print "%-10s" % pname[i],
        for j in range(i+1):
	    print "%10f" % (cov[i,j]/math.sqrt(cov[i,i]*cov[j,j]),),
        #endfor
        print
    #endfor

    print
    print
    print
    
    return p, rms

#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Parametric function:  
#  p is the parameter vector; 
def fp4(p,dmjd_array,airmass_array,pwv_array):
    return p[0] + p[1]*dmjd_array + p[2]*np.power(airmass_array,0.6) + p[3]*pwv_array

#--------------------------------------------------------------------------
# Error function:
def residuals4(p,dmjd_array,airmass_array,pwv_array,dmag_array):
    err = (dmag_array-fp4(p,dmjd_array,airmass_array,pwv_array))
    return err

#--------------------------------------------------------------------------
# Fitting code:
def aTmCamTestFit4(dmjd_array, airmass_array, pwv_array, dmag_array):

    # Calculate the median of dmag for use as an initial guess
    # for the overall zeropoint offset..
    mdn = np.median( dmag_array, None )

    # Parameter names
    pname = (['a_0', 'a_1', 'k_airmass', 'k_pwv'])

    # Initial parameter values
    p0 = [mdn, 0.0, 0.0, 0.0]

    print 
    print 'Initial parameter values:  ', p0

    #print fp4(p0,dmjd_array,airmass_array,pwv_array)
    #print residuals4(p0,dmjd_array,airmass_array,pwv_array,dmag_array)

    # Perform fit

    p,cov,infodict,mesg,ier = leastsq(residuals4, p0, args=(dmjd_array, airmass_array, pwv_array, dmag_array), maxfev=10000, full_output=1)

    if ( ier>=1 and ier <=4):
        print "Converged"
    else:
        print "Not converged"
        print mesg


    # Calculate some descriptors of the fit 
    # (similar to the output from gnuplot 2d fits)

    chisq=sum(infodict['fvec']*infodict['fvec'])
    dof=len(dmag_array)-len(p)
    rms=math.sqrt(chisq/dof)
    
    print "Converged with chi squared ",chisq
    print "degrees of freedom, dof ", dof
    print "RMS of residuals (i.e. sqrt(chisq/dof)) ", rms
    print "Reduced chisq (i.e. variance of residuals) ", chisq/dof
    print


    # uncertainties are calculated as per gnuplot, "fixing" the result
    # for non unit values of the reduced chisq.
    # values at min match gnuplot
    print "Fitted parameters at minimum, with 68% C.I.:"
    for i,pmin in enumerate(p):
        print "%-10s %13g +/- %13g   (%5f percent)" % (pname[i],pmin,math.sqrt(cov[i,i])*math.sqrt(chisq/dof),100.*math.sqrt(cov[i,i])*math.sqrt(chisq/dof)/abs(pmin))
    print


    print "Correlation matrix:"
    # correlation matrix close to gnuplot
    print "               ",
    for i in range(len(pname)): print "%-10s" % (pname[i],),
    print
    for i in range(len(p)):
        print "%-10s" % pname[i],
        for j in range(i+1):
	    print "%10f" % (cov[i,j]/math.sqrt(cov[i,i]*cov[j,j]),),
        #endfor
        print
    #endfor

    print
    print
    print
    
    return p, rms

#--------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#--------------------------------------------------------------------------
