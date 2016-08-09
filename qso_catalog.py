
import os, sys
import pylab
import numpy as np
import healpy as hp
import pandas as pd
import Get_files
import matplotlib.pyplot as plt

pd.set_option('display.mpl_style', 'default')

params1 = {'backend': 'pdf',
               'axes.labelsize': 15,
               'text.fontsize': 18,
               'xtick.labelsize': 18,
               'ytick.labelsize': 18,
               'legend.fontsize': 8,
               'lines.markersize': 16,
               'font.size': 16,}
pylab.rcParams.update(params1)


class Ini_params():
    def __init__(self):

        self.del_chisq = 4
        self.Npix_side = 2**5
        self.verbose   = False
        self.rep_thid  = 4
        self.direct    = 'data/'
        self.full_file = 'spAll-v5_10_0.fits'
        self.sub_file  = 'subset_spAll-v5_10_0.csv'

        self.bit_boss  = [10,11,12,13,14,15,16,17,18,19,40,41,42,43,44]
        self.bit_eboss = [10,11,12,13,14,15,16,17,18]

        self.targets   = {'BOSS_TARGET1': self.bit_boss, 'EBOSS_TARGET0': self.bit_eboss,
                                   'EBOSS_TARGET1': self.bit_eboss}

        self.condition = 'CLASS== "QSO".ljust(6) & (OBJTYPE=="QSO".ljust(16) | ' \
                         'OBJTYPE=="NA".ljust(16)) & THING_ID != -1'

        self.spall_cols = ['RA','DEC','THING_ID','MJD','PLATE','FIBERID','BOSS_TARGET1',
                           'EBOSS_TARGET0','EBOSS_TARGET1']

        self.spec_cols  = ['flux','loglam','ivar','and_mask','or_mask', 'wdisp', 'sky', 'model']


    def do_nothing(self):
        pass





class Qso_catalog():
    def __init__(self, df_fits, verbose = True):
        self.df_fits     = df_fits
        self.verbose     = verbose
        self.chisq_dist  = []
        #Ini_params.__init__(self)



    def searching_quasars(self, data_column, mask_bit):
        """Filter the quasar according to the bit array,
        return a Boolean array wherever the quasar is"""

        is_qso  =  lambda bit: ((self.df_fits[data_column] & 2**bit) > 0)
        all_qso =  map(is_qso, mask_bit)
        return reduce(lambda x, y: x | y, all_qso)



    def filtering_qsos(self, targets, condition):
        """Filter only the quasars withing the dataFrame"""

        # only those with CLASS=QSO & OBJTYPE=(QSO|NA)
        self.df_fits  = self.df_fits.query(condition)

        # and satisfy the bit condition
        a =[]
        for targ, bits in targets.iteritems():
            a.append(self.searching_quasars(targ, bits))
        self.df_qsos = self.df_fits[reduce(lambda x, y: x | y, a)].copy()


        print 'Total qsos (Bit_condition:Ok & %s='%(condition), len(self.df_qsos)
        return 0




    def adding_pixel_column(self, Npix_side):
        """Computing healpix pixel given 'DEC' and 'RA' """
        phi_rad   = lambda ra : ra*np.pi/180.
        theta_rad = lambda dec: (90.0 - dec)*np.pi/180.

        self.df_qsos['PIX'] = hp.ang2pix(Npix_side, theta_rad(self.df_qsos['DEC']), phi_rad(self.df_qsos['RA']))
        unique_pixels  = self.df_qsos['PIX'].unique()
        print 'Unique pixels: ', len(unique_pixels)

        #setting 'PIX' and 'THING_ID' as indices
        self.df_qsos = self.df_qsos.set_index(['PIX','THING_ID'], drop=False).drop('PIX', 1).sort_index()

        if self.verbose: print self.df_qsos.head()
        return unique_pixels




    def pix_uniqueid(self, pix_id, repetitions= 2):
        """Given a pixel, return the THING_ID and the number of times is repeated"""
        rep_thing_id = self.df_qsos.query('PIX == {}'.format(pix_id)).groupby('THING_ID').size()

        if not rep_thing_id[rep_thing_id >= repetitions].empty:
            uniqeid = dict(rep_thing_id[rep_thing_id >= repetitions])
        else:
            uniqeid = {}
        return uniqeid




    def get_names(self, thing_id, name):
        return list(self.df_qsos.query('THING_ID == %s'%(thing_id))[name].values)




    def get_files(self, direct, thing_id ='thing_id', passwd= None):
        plates   = self.get_names(thing_id, 'PLATE')
        mjds     = self.get_names(thing_id, 'MJD')
        fiberids = self.get_names(thing_id, 'FIBERID')
        plate_n  = ['{}'.format(plate) for plate in plates]

        qso_files= ['spec-%s-%s-%s'%(plate, mjd, str(fiberid).zfill(4))
                        for plate, mjd, fiberid in zip(plates, mjds, fiberids)]

        for plate, file in zip(plate_n, qso_files):
            file = '{}.fits'.format(file)
            if not os.path.isfile(direct + file):
                if passwd is None:
                    Get_files.get_bnl_files(direct, plate, file)
                else:
                    Get_files.get_web_files(direct, plate, file, passwd)
        return qso_files




    def stack_repeated(self, direct, qso_files, columns):
        stack_qsos = []
        for i, fqso in enumerate(qso_files):
            stack_qsos.append(Get_files.read_fits(direct, fqso, columns).set_index('loglam'))
            stack_qsos[i]['flux_%s'%(fqso)] = stack_qsos[i]['flux']
            stack_qsos[i]['ivar_%s'%(fqso)] = stack_qsos[i]['ivar']

        result   = pd.concat([stack_qsos[j][['flux_%s'%(stacks),'ivar_%s'%(stacks)]] for j, stacks in enumerate(qso_files)], axis=1)
        return result.fillna(0).copy()



    def coadds(self, direct, qso_files, columns):
        dfall_qsos = self.stack_repeated(direct, qso_files, columns)
        dfall_qsos['sum_flux_ivar'] =0
        dfall_qsos['sum_ivar']      =0
        for i, fqso in enumerate(qso_files):
            dfall_qsos['sum_flux_ivar'] += dfall_qsos['flux_%s'%(fqso)]*dfall_qsos['ivar_%s'%(fqso)]
            dfall_qsos['sum_ivar']      += dfall_qsos['ivar_%s'%(fqso)]

        dfall_qsos['coadd'] = dfall_qsos['sum_flux_ivar']/dfall_qsos['sum_ivar']

        dfall_qsos = dfall_qsos.fillna(0).copy()
        return dfall_qsos



    def calc_chisq(self, qso_files, dfall_qsos):
        """Compute chisq and return a dict with files'name and chisq"""
        chi_sq_all =[]
        for i, fqso in enumerate(qso_files):
            chis_sq = np.sum((dfall_qsos['coadd'].values - dfall_qsos['flux_%s'%(fqso)].values)**2*dfall_qsos['ivar_%s'%(fqso)].values)
            chi_sq_all.append(chis_sq/len(dfall_qsos.values))
        return dict(zip(qso_files, chi_sq_all))



    def select_chisq(self, zipchisq, del_chisq):
        tmp_zipchisq = zipchisq.copy()
        for files, chisq in zipchisq.iteritems():
            if chisq > del_chisq:
                del tmp_zipchisq[files]
            else: continue

        return tmp_zipchisq.keys()



    def plot_coadds(self, dfall_qsos, thingid, zipchisq):
        plt.figure(figsize = (18, 8))
        xlimits = [3.55, 4]
        ylimits = [-10, 25]
        ax = plt.subplot(1, 2, 1)
        for fqso, chisq in zipchisq.iteritems():
            dfall_qsos['flux_%s'%(fqso)].plot(label='%s  , chisq=%s'%(fqso, chisq),
                                         xlim=xlimits, ylim=ylimits, ax=ax)
        plt.legend(loc='best')

        ax2 = plt.subplot(1,2,2)
        dfall_qsos['coadd'].plot(label='coad', xlim=xlimits, ylim=ylimits, ax=ax2)
        plt.legend(loc='best')
        plt.title('THING_ID: %s'%(thingid))
        plt.show(block=True)
        return 0


    def plot_chisq_dist(self, zipchisq):
        for i in zipchisq.values():
            self.chisq_dist.append(i)

        plt.hist(self.chisq_dist, bins=100, range=(0,10))
        #plt.show(block=True)
        plt.ylabel('#')
        plt.xlabel('chisq')
        plt.title('chisq Histogram')
        plt.savefig('chisq.pdf')
        return 0



if __name__=='__main__':
    Pars      = Ini_params()


    #instead read the subset we're interested on
    df_fits   = Get_files.read_subset_fits(Pars.sub_file)
    Qsos      = Qso_catalog(df_fits, verbose= Pars.verbose)

    for targ, bits in Pars.targets.iteritems():
        print 'Quasars with Bit_condition:Ok in {} ='.format(targ), Qsos.searching_quasars(targ, bits).sum()


