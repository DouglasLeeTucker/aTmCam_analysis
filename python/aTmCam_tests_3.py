#!/usr/bin/env python

# Initial setup...
import numpy as np
import pandas as pd
from scipy import interpolate
import glob
import math
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import os


#--------------------------------------------------------------------------
# Main code.
def main():
    

    # Let's grab all the observing logs for all three 3 years into a single sorted list...
    obsLogList = sorted(glob.glob('aTmCam_data/Test201?/???????-????????/???????-????????_obs_log.txt'))

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

    # Let's rename column "#expid" to "expid"...
    obsdf.rename(columns={'#expid': 'expid'}, inplace=True)

    # Oops... somehow, expid became a "float"...  let's re-cast it back to an "int"...
    obsdf.expid = obsdf.expid.astype(int)


    # Let's now grab all the instrumental magnitude files for all three 3 years into a single 
    #  sorted list...
    magLogList = sorted(glob.glob('aTmCam_data/Test201?/???????-????????/???????-????????_final_data_bin2.txt'))

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
    
        # Let's plot dmag vs. MJD of the 3 main Hipparcos in one plot...
        ax = magobsdf[((magobsdf.object=='HIP42334') | (magobsdf.object=='HIP75501') | (magobsdf.object=='HIP117452'))].plot('mjd',dmag, grid=True, ylim=[-0.5,5.0])
        fig = ax.get_figure()
        outputFile = """./Temp/%s_vs_mjd_hipbig3.png""" % (dmag)
        fig.savefig(outputFile)
        plt.close()
        
        # Let's plot a 2D histogram of log(Nobs), binned by dmag and airmass, for HIP117452...
        x=magobsdf[(magobsdf.object=='HIP117452')]['airmass']
        y=magobsdf[(magobsdf.object=='HIP117452')][dmag]
        title = 'HIP117452'
        outputFile = """./Temp/%s_vs_airmass_logN.HIP117452.png""" % (dmag)
        aTmCam2DHistDensPlot(x,y,1.0,3.25,-0.3,0.8,title,'airmass',dmag,'log10(N)',outputFile)
        
        # Let's plot a 2D histogram of log(Nobs), binned by dmag and airmass, for the Big 3...
        x=magobsdf[((magobsdf.object=='HIP42334') | (magobsdf.object=='HIP75501') | (magobsdf.object=='HIP117452'))]['airmass']
        y=magobsdf[((magobsdf.object=='HIP42334') | (magobsdf.object=='HIP75501') | (magobsdf.object=='HIP117452'))][dmag]
        title = 'HIP117452, HIP75501, and HIP42334 (The Big 3)'
        outputFile = """./Temp/%s_vs_airmass_logN.big3.png""" % (dmag)
        aTmCam2DHistDensPlot(x,y,1.0,3.25,-0.3,0.8,title,'airmass',dmag,'log10(N)',outputFile)

        # Let's plot a 2D histogram of log(Nobs), binned by "dmag-dmag3" and airmass, for HIP117452...
        # ... but skip if dmag is dmag3...
        if dmag=='dmag3': continue
        x=magobsdf[(magobsdf.object=='HIP117452')]['airmass']
        y1=magobsdf[(magobsdf.object=='HIP117452')][dmag]
        y2=magobsdf[(magobsdf.object=='HIP117452')]['dmag3']
        y=y1-y2
        title = 'HIP117452'
        ylabel="""%s-dmag3""" % (dmag)
        outputFile = """./Temp/%s3_vs_airmass_logN.HIP117452.png""" % (dmag)
        aTmCam2DHistDensPlot(x,y,1.0,3.25,-0.3,0.8,title,'airmass',ylabel,'log10(N)',outputFile)

        # Let's plot a 2D histogram of log(Nobs), binned by "dmag-dmag3" and airmass, for the Big 3...
        # ... but skip if dmag is dmag3...
        if dmag=='dmag3': continue

        x=magobsdf[((magobsdf.object=='HIP42334') | (magobsdf.object=='HIP75501') | (magobsdf.object=='HIP117452'))]['airmass']
        y1=magobsdf[((magobsdf.object=='HIP42334') | (magobsdf.object=='HIP75501') | (magobsdf.object=='HIP117452'))][dmag]
        y2=magobsdf[((magobsdf.object=='HIP42334') | (magobsdf.object=='HIP75501') | (magobsdf.object=='HIP117452'))]['dmag3']
        y=y1-y2        
        title = 'HIP117452, HIP75501, and HIP42334 (The Big 3)'
        ylabel="""%s-dmag3""" % (dmag)
        outputFile = """./Temp/%s3_vs_airmass_logN.big3.png""" % (dmag)
        aTmCam2DHistDensPlot(x,y,1.0,3.25,-0.3,0.8,title,'airmass',ylabel,'log10(N)',outputFile)

    # endfor


    # Precipitable Water Vapor (PWV)

    # The Suominet-processed GPSmon data are in one big CSV file, updated daily...
    gpsdatafile = 'GPSmon_data/SuominetResults.csv'

    # Read it into a pandas DataFrame
    # Note, however, the separator is not just a comma, but a comma-plus-a-single-whitespace,
    # so we need to note the separator explicitly in the pandas read_csv command...
    gpsdf = pd.read_csv(gpsdatafile, sep=', ')

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
    title = 'HIP117452'
    outputFile = "./Temp/dmag4_vs_pwv_logN.HIP117452.png"
    aTmCam2DHistDensPlot(x,y,0.0,12.0,-0.3,0.8,title,'PWV','dmag4','log10(N)',outputFile)

    # If we plot dmag4-dmag3, we can remove the effects of clouds and other "gray" throughput variations...
    x=magobsdf_new[(magobsdf_new.object=='HIP117452')].pwv
    y1=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag4
    y2=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag3
    y=y1-y2
    title = 'HIP117452'
    outputFile = "./Temp/dmag43_vs_pwv_logN.HIP117452.png"
    aTmCam2DHistDensPlot(x,y,0.0,12.0,-0.3,0.8,title,'PWV','dmag4-dmag3','log10(N)',outputFile)

    # This is like the original delta_mag4 vs. airmass plot, but now color-coded by PWV...
    x=magobsdf_new[(magobsdf_new.object=='HIP117452')].airmass
    y=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag4
    z=magobsdf_new[(magobsdf_new.object=='HIP117452')].pwv
    title = 'HIP117452'
    outputFile = "./Temp/dmag4_vs_airmass_pwv.HIP117452.png"
    aTmCam2DHistXYZPlot(x,y,z,1.0,3.25,-0.3,0.8,title,'airmass','dmag4','PWV',outputFile)
    
    # Same as above, but now dmag4-dmag3 vs. airmass, color-coded by PWV...
    x=magobsdf_new[(magobsdf_new.object=='HIP117452')].airmass
    y1=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag4
    y2=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag3
    y=y1-y2
    z=magobsdf_new[(magobsdf_new.object=='HIP117452')].pwv
    title = 'HIP117452'
    outputFile = "./Temp/dmag43_vs_airmass_pwv.HIP117452.png"
    aTmCam2DHistXYZPlot(x,y,z,1.0,3.25,-0.8,0.8,title,'airmass','dmag4-dmag3','PWV',outputFile)

    # Let's switch things around a bit, plotting dmag4 vs. PWV and color-coding by airmass...
    z=magobsdf_new[(magobsdf_new.object=='HIP117452')].airmass
    y=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag4
    x=magobsdf_new[(magobsdf_new.object=='HIP117452')].pwv
    title = 'HIP117452'
    outputFile = "./Temp/dmag4_vs_pwv_airmass.HIP117452.png"
    aTmCam2DHistXYZPlot(x,y,z,0.0,12.0,-0.3,0.8,title,'PWV','dmag4','airmass',outputFile)

    # Same as two figures above, but switching PWV and airmass axes...
    x=magobsdf_new[(magobsdf_new.object=='HIP117452')].pwv
    y1=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag4
    y2=magobsdf_new[(magobsdf_new.object=='HIP117452')].dmag3
    y=y1-y2
    z=magobsdf_new[(magobsdf_new.object=='HIP117452')].airmass
    title = 'HIP117452'
    outputFile = "./Temp/dmag43_vs_pwv_airmass.HIP117452.png"
    aTmCam2DHistXYZPlot(x,y,z,0.0,12.0,-0.8,0.8,title,'PWV','dmag4-dmag3','airmass',outputFile)

    #magobsdf_new.to_csv('/Users/dtucker/Desktop/magobsdf_new.csv', index=False)

    
    #
    # Let's work a bit with just the HIP117452 data over the different optics cleaning epochs experienced by aTmCam...
    #

    # Read in the aTmCam "epochs" file...
    # Grab path and name of aTmCam_epochs.csv file in the DECam_PGCM data directory...
    #  Is there a better way to do this?  See also:
    #  http://stackoverflow.com/questions/779495/python-access-data-in-package-subdirectory
    # Absolute path for the directory containing this module:
    moduleAbsPathName = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aTmCamEpochsFile = os.path.join(moduleAbsPathName, "data", "aTmCam_epochs.csv")
    if os.path.isfile(aTmCamEpochsFile)==False:
        print """aTmCamEpochsFile %s does not exist...""" % (aTmCamEpochsFile)
        print 'Returning with error code 1 now...'
        return 1
    aTmCamEpochsDF = pd.read_csv(aTmCamEpochsFile, comment='#')
    print 'aTmCamEpochs:  '
    print aTmCamEpochsDF
    nepochs = aTmCamEpochsDF.epoch.size
    
    for iepoch in range(nepochs):

        epoch = aTmCamEpochsDF.epoch.iloc[iepoch]
        mjd_start = aTmCamEpochsDF.mjd_start.iloc[iepoch]
        mjd_end = aTmCamEpochsDF.mjd_end.iloc[iepoch]

        print """Epoch:  %s  (%d <= MJD < %d)""" % (epoch, mjd_start, mjd_end)
    
        # We will call the new dataframe df...
        df_orig = magobsdf_new[( (magobsdf_new.mjd >= mjd_start) &
                                 (magobsdf_new.mjd < mjd_end) &
                                 (magobsdf_new.object=='HIP117452') )].copy()
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

            # Create QA plots for this iteration...
            status = aTmCamCloudFitQA(df, p, epoch, iiter)

        # endfor
    
        df.drop('vigintile', axis=1, inplace=True)

        # Add a dmjd column to df...
        mjd0 = int(df.mjd.min()+0.3)
        df.loc[:,'dmjd'] = df.loc[:,'mjd'] - mjd0


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

                # Create QA plots for this iteration...
                status = aTmCamDmagFitQA(df, dmag, epoch, iiter)

            # endfor
    
            # Create 2D histogram QA plots and output a CSV file of residuals...
            status = aTmCam2DHistQA(df, dmag, epoch)

        # endfor
        
        # Let's try the above for (mag1,mag2,mag3), but solving for k on a night-by-night basis... 

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

                # Update/create nightIndex column...
                # If there is a pre-existing nightIndex column in df, delete it...
                if 'nightIndex' in df.columns:
                    df.drop('nightIndex', axis=1, inplace=True)
                # Create nightIndex column for df...
                nightArray = np.sort(df.imjd.unique())
                nightDF = pd.DataFrame(nightArray, columns=['imjd'])
                nightDF['nightIndex'] = nightDF.index
                df = pd.merge(df, nightDF, on='imjd', how='inner')

                outCorrFile = """./Temp/%s_corrmatrix_hip117542_mjdfit-nightly-k_%s_iter%d.fits""" % (dmag, epoch, iiter)
                p,perr,pname,rms = aTmCamTestFitNightlyK(df, nightDF, dmag, outCorrFile)
                df.loc[:,'res'] = residualsNightlyK(p,df.loc[:,'dmjd'],df.loc[:,'nightIndex'],df.loc[:,'airmass'],df.loc[:,dmag])

                # Output parameter fits
                mjd0Array = len(nightArray)*[mjd0]
                a0Array = len(nightArray)*[p[0]]
                a0errArray = len(nightArray)*[perr[0]]
                a1Array = len(nightArray)*[p[1]]
                a1errArray = len(nightArray)*[perr[1]]
                rmsArray = len(nightArray)*[rms]
                resultsDF = pd.DataFrame(
                    {'imjd': nightArray,
                     'a0': a0Array,
                     'a0_err': a0errArray,
                     'a1': a1Array, 
                     'a1_err': a1errArray,
                     'mjd0' : mjd0Array, 
                     'pname': pname[2:],
                     'k': p[2:],
                     'k_err': perr[2:],
                     'rms': rmsArray
                    })
                # Re-order columns:
                resultsDF = resultsDF[['imjd','mjd0','a0','a0_err','a1','a1_err','k','k_err','rms','pname']]
                # Actual output...
                outputFile = """./Temp/%s_results_hip117542_mjdfit-nightly-k_%s_iter%d.csv""" % (dmag, epoch, iiter)
                resultsDF.to_csv(outputFile,index=False)
            
                stddev = df['res'].std()
                mask = (np.abs(df.res)< nsigma*stddev)

                # Create QA plots for this iteration...
                status = aTmCamDmagFitQA(df, dmag, epoch, iiter, '-nightly-k')

            # endfor

            # Create 2D histogram QA plots and output a CSV file of residuals...
            status = aTmCam2DHistQA(df, dmag, epoch, '.nightly-k')

        # endfor

        
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

            # Create QA plots for this iteration...
            status = aTmCamDmagFitQA(df, dmag, epoch, iiter)

        # endfor
    
        # Create 2D histogram QA plots and output a CSV file of residuals...
        status = aTmCam2DHistQA(df, dmag, epoch)
            

        # Now we work on dmag4-dmag3...
        dmag = 'dmag43'
        print dmag

        df = df_orig_clean.copy()
        df['dmag43'] = df['dmag4']-df['dmag3']
    
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

            p,rms = aTmCamTestFit43(df.loc[:,'dmjd'], df.loc[:,'airmass'], df.loc[:,'pwv'], df.loc[:,dmag])
            df.loc[:,'res'] = residuals43(p,df.loc[:,'dmjd'],df.loc[:,'airmass'],df.loc[:,'pwv'],df.loc[:,dmag])

            stddev = df['res'].std()
            mask = (np.abs(df.res)< nsigma*stddev)
    
            # Create QA plots for this iteration...
            status = aTmCamDmagFitQA(df, dmag, epoch, iiter)

        # endfor

        # Create 2D histogram QA plots and output a CSV file of residuals...
        status = aTmCam2DHistQA(df, dmag, epoch)
            
    # endfor(epochs)

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
# Parametric function:  
#  p is the parameter vector; 
def fp43(p,dmjd_array,airmass_array,pwv_array):
    #return p[0] + p[1]*dmjd_array + p[2]*np.power(airmass_array,0.6) + p[3]*pwv_array
    return p[0] + p[1]*dmjd_array + p[2]*airmass_array + p[3]*pwv_array + p[4]*pwv_array**2
    #return p[0] + 0.00*dmjd_array + p[1]*airmass_array + p[2]*pwv_array + p[3]*pwv_array**2
    #return p[0] + 0.00*dmjd_array + p[1]*airmass_array + 0.00*pwv_array + p[2]*pwv_array*pwv_array
    #return p[0] + 0.00*dmjd_array + p[1]*airmass_array + 0.00*pwv_array + p[2]*np.power(pwv_array,4.0)

#--------------------------------------------------------------------------
# Error function:
def residuals43(p,dmjd_array,airmass_array,pwv_array,dmag_array):
    err = (dmag_array-fp43(p,dmjd_array,airmass_array,pwv_array))
    return err

#--------------------------------------------------------------------------
# Fitting code:
def aTmCamTestFit43(dmjd_array, airmass_array, pwv_array, dmag_array):

    # Calculate the median of dmag for use as an initial guess
    # for the overall zeropoint offset..
    mdn = np.median( dmag_array, None )

    # Parameter names
    #pname = (['a_0', 'a_1', 'k_airmass', 'k_pwv'])
    pname = (['a_0', 'a_1', 'k_airmass', 'k1_pwv', 'k2_pwv'])
    #pname = (['a_0', 'k_airmass', 'k1_pwv', 'k2_pwv'])
    #pname = (['a_0', 'k_airmass', 'k2_pwv'])
    #pname = (['a_0', 'k_airmass', 'k4_pwv'])

    # Initial parameter values
    #p0 = [mdn, 0.0, 0.0, 0.0]
    p0 = [mdn, 0.0, 0.0, 0.0, 0.0]
    #p0 = [mdn, 0.0, 0.0, 0.0]
    #p0 = [mdn, 0.0, 0.0]
    #p0 = [mdn, 0.0, 0.0]

    print 
    print 'Initial parameter values:  ', p0

    #print fp43(p0,dmjd_array,airmass_array,pwv_array)
    #print residuals43(p0,dmjd_array,airmass_array,pwv_array,dmag_array)

    # Perform fit

    p,cov,infodict,mesg,ier = leastsq(residuals43, p0, args=(dmjd_array, airmass_array, pwv_array, dmag_array), maxfev=10000, full_output=1)

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
# Parametric function:  
#  p is the parameter vector; 
def fpNightlyK(p,dmjd_array,night_index_array,airmass_array):
    return p[0] + p[1]*dmjd_array + p[2+night_index_array]*airmass_array

#--------------------------------------------------------------------------
# Error function:
def residualsNightlyK(p,dmjd_array,night_index_array,airmass_array,dmag_array):
    err = (dmag_array-fpNightlyK(p,dmjd_array,night_index_array,airmass_array))
    return err

#--------------------------------------------------------------------------
# Fitting code:
def aTmCamTestFitNightlyK(df, nightDF, dmag, outCorrFile=None):

    # Only need astropy.io.fits if we are outputting an outCorrFile...
    if outCorrFile is not None:
        from astropy.io import fits
    

    # Extract arrays needed later...
    # (dmjd_array, night_index_array, airmass_array, dmag_array
    dmjd_array = df.loc[:,'dmjd']
    night_index_array = df.loc[:,'nightIndex']
    airmass_array = df.loc[:,'airmass']
    dmag_array = df.loc[:,dmag]

    
    # Calculate the median of dmag for use as an initial guess
    # for the overall zeropoint offset..
    mdn = np.median( dmag_array, None )


    # Set the parameter names and the initial parameter values...
    pname = (['a_0', 'a_1'])
    p0 = [mdn, 0.0]
    for ii in range(nightDF.imjd.size):
        kname = 'k_'+str(nightDF.loc[ii, 'imjd'])
        #kname = 'k_'+str(nightDF.loc[ii, 'nightIndex'])
        pname.append(kname)
        p0.append(0.)

    print 
    print 'Initial parameter values:  ', p0


    # Perform fit

    p,cov,infodict,mesg,ier = leastsq(residualsNightlyK, p0, args=(dmjd_array, night_index_array, airmass_array, dmag_array), maxfev=10000, full_output=1)

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
    perr = []
    for i,pmin in enumerate(p):
        print "%-10s %13g +/- %13g   (%5f percent)" % (pname[i],pmin,math.sqrt(cov[i,i])*math.sqrt(chisq/dof),100.*math.sqrt(cov[i,i])*math.sqrt(chisq/dof)/abs(pmin))
        perr.append(math.sqrt(cov[i,i])*math.sqrt(chisq/dof))
    print


    print "Correlation matrix (up to first 10 parameters in parameter list):"
    nsize1 = min(len(pname), 10)
    nsize2 = min(len(p), 10)
    # correlation matrix close to gnuplot
    print "               ",
    for i in range(nsize1): print "%-10s" % (pname[i],),
    print
    for i in range(nsize2):
        print "%-10s" % pname[i],
        for j in range(i+1):
	    print "%10f" % (cov[i,j]/math.sqrt(cov[i,i]*cov[j,j]),),
        #endfor
        print
    #endfor

    if outCorrFile is not None:
        # Create a numpy array "corr" with the same shape as the numpy array "cov",
        #  update all its entries with the values of the correlation matrix, and
        #  write the result to a FITS image...
        corr = np.copy(cov)
        for i in range(len(p)):
            for j in range(len(p)):
	        corr[i,j] = cov[i,j]/math.sqrt(cov[i,i]*cov[j,j])
            # endfor
        # endfor
        hdu = fits.PrimaryHDU(corr)
        hdu.writeto(outCorrFile,clobber=True)
    
    print
    print
    print
    
    return p, perr, pname, rms


#--------------------------------------------------------------------------

def aTmCamSimpleXYPlotWithFitLine(df,col_x,col_y,p,outputFile):
    
    ax=df.plot(col_x,col_y, grid=True, kind='scatter')
    plt.plot(df[col_x],p(df[col_x]),'m-')
    fig = ax.get_figure()
    fig.savefig(outputFile)
    plt.close()

    return 0


#--------------------------------------------------------------------------

def aTmCamSimpleXYPlot(df,col_x,col_y,outputFile):
    
    ax=df.plot(col_x,col_y, grid=True, kind='scatter')
    fig = ax.get_figure()
    fig.savefig(outputFile)
    plt.close()

    return 0


#--------------------------------------------------------------------------

def aTmCamSimpleHistPlot(df,col,outputFile):
    
    #ax=df.hist(col, grid=True, bins=100)
    ax=df[col].hist(grid=True, bins=100)
    fig = ax.get_figure()
    fig.savefig(outputFile)
    plt.close()

    return 0


#--------------------------------------------------------------------------

def aTmCamCloudFitQA(df, p, epoch, iiter):

    # Plot dmag3 vs. MJD...
    outputFile = """./Temp/dmag3_vs_mjd_hip117542_cloudcull_%s_iter%d.png""" % (epoch, iiter)
    status = aTmCamSimpleXYPlotWithFitLine(df,'mjd','dmag3',p,outputFile)
    
    # Plot histogram of dmag3 residuals...
    outputFile = """./Temp/dmag3_vs_mjd_hip117542_cloudcull_reshist_%s_iter%d.png""" % (epoch, iiter)
    status = aTmCamSimpleHistPlot(df,'res',outputFile)
    
    return 0


#--------------------------------------------------------------------------

def aTmCamDmagFitQA(df, dmag, epoch, iiter, extra_label=''):

    # Plot residuals vs. MJD...
    outputFile = """./Temp/%sres_vs_mjd_hip117542_mjdfit%s_%s_iter%d.png""" % (dmag, extra_label, epoch, iiter)
    status = aTmCamSimpleXYPlot(df,'mjd','res',outputFile)

    # Plot residuals vs. airmass...
    outputFile = """./Temp/%sres_vs_airmass_hip117542_mjdfit%s_%s_iter%d.png""" % (dmag, extra_label, epoch, iiter)
    status = aTmCamSimpleXYPlot(df,'airmass','res',outputFile)
    
    # Plot residuals vs. PWV (mostly needed by band4)...
    outputFile = """./Temp/%sres_vs_pwv_hip117542_mjdfit_%s_iter%d.png""" % (dmag, epoch, iiter)
    status = aTmCamSimpleXYPlot(df,'pwv','res',outputFile)

    # Plot histogram of dmag residuals...
    outputFile = """./Temp/%sres_vs_mjd_hip117542_mjdfit_reshist%s_%s_iter%d.png""" % (dmag, extra_label, epoch, iiter)
    status = aTmCamSimpleHistPlot(df,'res',outputFile)

    return 0


#--------------------------------------------------------------------------

def aTmCam2DHistDensPlot(x,y,xmin,xmax,ymin,ymax,title,xlabel,ylabel,cblabel,outputFile):
    
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, gridsize=100, bins='log', cmap='inferno')
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(cblabel)
    plt.grid(True)
    ax.grid(color='white')
    fig.savefig(outputFile)
    plt.close()
    
    return 0


#--------------------------------------------------------------------------
def aTmCam2DHistXYZPlot(x,y,z,xmin,xmax,ymin,ymax,title,xlabel,ylabel,cblabel,outputFile):
    
    fig, axs = plt.subplots(ncols=1)
    ax=axs
    hb = ax.hexbin(x, y, C=z, gridsize=100, cmap='rainbow', reduce_C_function=np.median)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(cblabel)
    plt.grid(True)
    ax.grid(color='white')
    fig.savefig(outputFile)
    plt.close()

    return 0
    

#--------------------------------------------------------------------------

def aTmCam2DHistQA(df, dmag, epoch, extra_label=''):

    # Let's plot a 2D histogram of log(Nobs), binned by dmagres and airmass, for HIP117452...
    title = 'HIP117452'
    ylabel = """%sres"""  % (dmag)
    outputFile = """./Temp/%sres_vs_airmass_logN.HIP117452.%s%s.png""" % (dmag, epoch, extra_label)
    aTmCam2DHistDensPlot(df['airmass'],df['res'],
                         1.0,3.25,
                         -3.0*df.res.std(),3.0*df.res.std(),
                         title,'airmass',ylabel,'log10(N)',
                         outputFile)

    
    # Let's plot a 2D histogram of PWV, binned by dmagres and airmass, for HIP117452...
    title = 'HIP117452'
    ylabel="""%sres"""  % (dmag)
    outputFile = """./Temp/%sres_vs_airmass_pwv.HIP117452.%s%s.png"""% (dmag, epoch, extra_label)
    aTmCam2DHistXYZPlot(df['airmass'],df['res'],df['pwv'],
                        1.0,3.25,
                        -3.0*df.res.std(),3.0*df.res.std(),
                        title,'airmass',ylabel,'PWV',
                        outputFile)


    # Let's plot a 2D histogram of log(Nobs), binned by dmagres and mjd, for HIP117452...
    title = 'HIP117452'
    ylabel = """%sres"""  % (dmag)
    outputFile = """./Temp/%sres_vs_mjd_logN.HIP117452.%s%s.png"""% (dmag, epoch, extra_label)
    aTmCam2DHistDensPlot(df['mjd'],df['res'],
                         df.mjd.min(),df.mjd.max(),
                         -3.0*df.res.std(),3.0*df.res.std(),
                         title,'MJD',ylabel,'log10(N)',
                         outputFile)
    
    
    # Let's plot a 2D histogram of PWV, binned by dmagres and mjd, for HIP117452...
    title = 'HIP117452'
    ylabel="""%sres"""  % (dmag)
    outputFile = """./Temp/%sres_vs_mjd_pwv.HIP117452.%s%s.png"""% (dmag, epoch, extra_label)
    aTmCam2DHistXYZPlot(df['mjd'],df['res'],df['pwv'],
                        df.mjd.min(),df.mjd.max(),
                        -3.0*df.res.std(),3.0*df.res.std(),
                        title,'MJD',ylabel,'PWV',
                        outputFile)


    # Let's plot a 2D histogram of airmass, binned by dmagres and mjd, for HIP117452...
    title = 'HIP117452'
    ylabel="""%sres"""  % (dmag)
    outputFile = """./Temp/%sres_vs_mjd_airmass.HIP117452.%s%s.png"""% (dmag, epoch, extra_label)
    aTmCam2DHistXYZPlot(df['mjd'],df['res'],df['airmass'],
                        df.mjd.min(),df.mjd.max(),
                        -3.0*df.res.std(),3.0*df.res.std(),
                        title,'MJD',ylabel,'airmass',
                        outputFile)


    # Let's plot a 2D histogram of log(Nobs), binned by dmagres and pwv, for HIP117452...
    title = 'HIP117452'
    ylabel = """%sres"""  % (dmag)
    outputFile = """./Temp/%sres_vs_pwv_logN.HIP117452.%s%s.png"""% (dmag, epoch, extra_label)
    aTmCam2DHistDensPlot(df['pwv'],df['res'],
                         df.pwv.min(),df.pwv.max(),
                         -3.0*df.res.std(),3.0*df.res.std(),
                         title,'PWV',ylabel,'log10(N)',
                         outputFile)


    # Output CSV file of residuals...
    outputFile = """Temp/df_%sres.HIP117452.%s%s.csv""" % (dmag, epoch, extra_label)
    df.to_csv(outputFile, index=False)    


    return 0


#--------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#--------------------------------------------------------------------------
