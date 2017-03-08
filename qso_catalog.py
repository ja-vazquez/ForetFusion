
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from get_files import *
from functools import reduce
from collections import defaultdict
import seaborn as sns
import fitsio

pd.set_option('display.mpl_style', 'default')

params1 = {'backend': 'pdf',
               'axes.labelsize': 15,
               'text.fontsize': 18,
               'xtick.labelsize': 18,
               'ytick.labelsize': 18,
               'legend.fontsize': 8,
               'lines.markersize': 16,
               'font.size': 16,}
#import pylab
#pylab.rcParams.update(params1)


class Ini_params():
    def __init__(self):

        self.trim_chisq  = 2                              #Thresold to discriminate from coadds
        self.rep_thid    = 4                              #Times we want a THING_ID repeated
        self.Npix_side   = 2**6                           #Nside to compute healpix

        self.write_master= False                          #Write master file with info
        self.write_ffits = False                          #Write fits files for each pix
        self.write_hist  = False                          #Write chisq distribution files
        self.write_names = False                          #Write names of all spec.fits files used
        self.show_plots  = False                          #must be False when using mpi
        self.need_files  = False                          #If files needed get them from bnl-cluster

        self.dir_spec    = 'data/spectra/'
        self.dir_v5_10   = 'v5_10_0/spectra/'
        self.pix_dir     = 'healpix/'

        self.full_file   = 'spAll-v5_10_0.fits'
        self.Spall_files = 'SpAll_files.csv'

        self.specs_dist  = 'specs_dist/specs_dist'
        self.specs_bad   = 'specs_dist/specs_bad'

        self.bit_boss    = [10,11,12,13,14,15,16,17,18,19,40,41,42,43,44]
        self.bit_eboss   = [10,11,12,13,14,15,16,17,18]

        self.targets     = {'BOSS_TARGET1': self.bit_boss, 'EBOSS_TARGET0': self.bit_eboss,
                                   'EBOSS_TARGET1': self.bit_eboss}

        self.condition   = 'CLASS== "QSO".ljust(6) & (OBJTYPE=="QSO".ljust(16) | ' \
                           'OBJTYPE=="NA".ljust(16)) & THING_ID != -1'

        self.spall_cols  = ['RA','DEC','THING_ID','MJD','PLATE','FIBERID','BOSS_TARGET1','CLASS','OBJTYPE',
                            'EBOSS_TARGET0','EBOSS_TARGET1','Z','Z_ERR','ZWARNING']

        self.spec_cols   = ['flux','loglam','ivar','and_mask','or_mask', 'wdisp', 'sky', 'model']

        self.bnl_dir     = 'astro:/data/boss/v5_10_0/spectra/'


    def do_nothing(self):
        pass




class Qso_catalog(Ini_params):
    def __init__(self, df_fits, verbose = True):
        self.df_fits     = df_fits
        self.verbose     = verbose
        self.df_qsos     = None         #Main DataFrame that contains all Qso info.
        self.all_info    = []
        Ini_params.__init__(self)




    def searching_quasars(self, data_column, mask_bit):
        """Filter quasars according to the bit array,
        return a Boolean array wherever the quasar is"""
        is_qso  =  lambda bit: ((self.df_fits[data_column] & 2**bit) > 0)
        all_qso =  map(is_qso, mask_bit)
        return reduce(lambda x, y: x | y, all_qso)




    def print_filter_qsos(self, df, text):
        """Print # of Quasars according to the filtering condtion"""
        for targ, bits in self.targets.items():
            print ("Quasars with {} condition in {} =".format(text, targ),
                   self.searching_quasars(targ, bits).sum())
        print ("Total qsos = {} \n ------------------".format(len(df)))


    def own_filter(self):
        #convert bits to strings
        str_objs = self.df_fits.dtypes[self.df_fits.dtypes == np.object].index.values
        for str_obj in str_objs:
            self.df_fits[str_obj] = self.df_fits[str_obj].str.decode("utf-8")

        self.df_qsos = self.df_fits.query('THING_ID != -1 & THING_ID != 0')
        return 0



    def filtering_qsos(self, condition):
        """Filter quasars given the condition"""

        if self.verbose: self.print_filter_qsos(self.df_fits, 'Bit')

        #convert bits to strings
        str_objs = self.df_fits.dtypes[self.df_fits.dtypes == np.object].index.values
        for str_obj in str_objs:
            self.df_fits[str_obj] = self.df_fits[str_obj].str.decode("utf-8")

        # only those with CLASS=QSO & OBJTYPE=(QSO|NA)
        self.df_fits  = self.df_fits.query(condition)

        if self.verbose: self.print_filter_qsos(self.df_fits, 'General')
            # and satisfy the bit condition (is a quasar)
        a =[]
        for targ, bits in self.targets.items():
            a.append(self.searching_quasars(targ, bits))
        self.df_qsos = self.df_fits[reduce(lambda x, y: x | y, a)].copy()
        
        self.df_qsos = self.df_qsos[self.df_qsos['Z']> 2]
	print self.df_qsos.head()
        if self.verbose: self.print_filter_qsos(self.df_qsos, 'Both')
        return 0




    def adding_pixel_column(self):
        """Computing healpix pixel given 'DEC' and 'RA', and return unique healpix """

        phi_rad   = lambda ra : ra*np.pi/180.
        theta_rad = lambda dec: (90.0 - dec)*np.pi/180.

        self.df_qsos['PIX'] = hp.ang2pix(self.Npix_side, theta_rad(self.df_qsos['DEC']), phi_rad(self.df_qsos['RA']))
        unique_pixels  = self.df_qsos['PIX'].unique()
        print ('Unique pixels: ', len(unique_pixels))

            #setting 'PIX' and 'THING_ID' as indices
        self.df_qsos = self.df_qsos.set_index(['PIX','THING_ID'], drop=False).drop('PIX', 1).sort_index()
        if self.verbose: print (self.df_qsos.head())

        return unique_pixels





    def pix_uniqueid(self, pix_id):
        """Given a pixel, return the THING_ID and the number of times is repeated"""

        rep_thing_id = self.df_qsos.query('PIX == {}'.format(pix_id)).groupby('THING_ID').size()
        #if not rep_thing_id[rep_thing_id >= self.rep_thid].empty: -> uniqeid = {}
        uniqeid = dict(rep_thing_id[rep_thing_id >= self.rep_thid])
        return uniqeid





    def get_bnl_files(self, plate, file_name):
        """nasty hack to get bnl-files, but will change it later"""

        bnl_folder = os.path.join(self.dir_spec, plate)

        if self.verbose: print ('Getting file {} from the bnl'.format(file_name))
        if not os.path.isdir(bnl_folder): os.system('mkdir {}'.format(bnl_folder))
        os.system('scp {0} {1}'.format(os.path.join(self.bnl_dir, file_name), bnl_folder))
        return 0





    def get_files(self, thing_id ='thing_id'):
        """Given a THING_ID locate names of the files,
            if we dont have the files, get them."""

        self.th_id  = thing_id
        spall_names = ['PLATE', 'MJD', 'FIBERID', 'Z', 'Z_ERR', 'ZWARNING']

        get_names   = lambda x : list(self.df_qsos.query('THING_ID == %s'%(self.th_id))[x].values)
        plates, mjds, fiberids, z, zerr, zwarning = list(map(get_names, spall_names))

        name_file  = lambda p,m,f: '{0}/spec-{0}-{1}-{2}.fits'.format(p, m, str(f).zfill(4))
        qso_files  = list(map(name_file, plates, mjds, fiberids))

        spall_val  = list(zip(plates, mjds, fiberids, z, zerr, zwarning))
        dict_qso   = dict(zip(qso_files, spall_val))

        #just in case we need and dont have the files stored, get them from the bnl-cluster
        if self.need_files:
            plates = list(map(str, plates))
            for plate, file in zip(plates, qso_files):
                if not os.path.isfile(self.dir_spec + file): self.get_bnl_files(plate, file)

        return dict_qso






    def write_file_names(self):
        """ Write a file that contains all names.fits for filtered quasars,
        so we can get them later from NERSC """

        fnames = 'fnames'
        print ('... Printing file-names in {}'.format(self.Spall_files))

        self.df_qsos[fnames] =  self.df_qsos['PLATE'].astype(str) + '/spec-' + \
                self.df_qsos['PLATE'].astype(str) + '-' + self.df_qsos['MJD'].astype(str)+ '-' + \
                self.df_qsos['FIBERID'].astype(str).str.zfill(4) + '.fits'

        with open(self.Spall_files, 'w') as f:
             for name in (self.dir_v5_10 + self.df_qsos[fnames]).values:
                 f.write(name + '\n')
        return 0





    def cal_chisq(self, dict_qso):
        """Check whether the chisq -with respect to zero flux- is more than 2,
         otherwise reject that observation"""
        dict_file  = {}; dict_chisq = {}

        comp_chisq = lambda x, y: np.sum(x**2 *y)/len(x)

        for fqso in list(dict_qso.keys()):
            df_file = read_fits(self.dir_spec , fqso, self.spec_cols).set_index('loglam')
            chisq   = comp_chisq(df_file['flux'].values, df_file['ivar'].values)

            if chisq > self.trim_chisq:
                dict_file[fqso]  = df_file
                dict_chisq[fqso] = chisq
            else:
                del dict_qso[fqso]
                if self.write_hist: self.write_stats_file('bad', self.th_id, fqso, chisq)
        return dict_file, dict_chisq, dict_qso





    def coadds(self, dict_file):
        """ Add coadd, and_mask and or_mask columns for a given THING_ID """

        flux, ivar   = 'flux_%s', 'ivar_%s'
        or_mask      = 'or_mask_%s'
        and_mask     = 'and_mask_%s'

        for fqso in list(dict_file.keys()):
            columns = {'flux': flux%(fqso), 'ivar': ivar%(fqso), 'or_mask': or_mask%(fqso), 'and_mask': and_mask%(fqso)}
            dict_file[fqso].rename(columns= columns, inplace=True)

        df_coadd = pd.concat(dict_file.values(), axis=1).fillna(0)
        df_coadd['flux_ivar'] = 0
        df_coadd['ivar']      = 0

        for _, fqso in enumerate(dict_file.keys()):
            df_coadd['flux_ivar']    += df_coadd[flux%(fqso)]*df_coadd[ivar%(fqso)]
            df_coadd['ivar']         += df_coadd[ivar%(fqso)]

        df_coadd['and_mask'] = (reduce(lambda x, y: x & y, [df_coadd[and_mask%(i)].astype(int) for i in dict_file.keys()]))
        df_coadd['or_mask']  = (reduce(lambda x, y: x | y, [df_coadd[or_mask%(i)].astype(int)  for i in dict_file.keys()]))
        df_coadd['coadd']    = df_coadd['flux_ivar'] / df_coadd['ivar']

        self.df_coadd = df_coadd.fillna(0)
        return self.df_coadd[['coadd', 'ivar', 'and_mask', 'or_mask']]





    def plot_coadds(self, dic_chisq):
        """Plot the spectra and coadds"""

        xlimits = [3.55, 4]; ylimits = [-10, 25]
        plt.figure(figsize = (18, 8))
        ax = plt.subplot(1, 2, 1)
        for fqso in dic_chisq.keys():
            self.df_coadd['flux_%s'%(fqso)].plot(label='%s  , Chisq=%s'%(fqso.replace('.fits',''), dic_chisq[fqso]),
                         xlim=xlimits, ylim=ylimits, ax=ax)
        plt.legend(loc='best')

        ax2 = plt.subplot(1, 2, 2)
        self.df_coadd['coadd'].plot(label='coadd_%s'%(self.th_id), xlim=xlimits, ylim=ylimits, ax=ax2)
        plt.legend(loc='best')
        plt.title('THING_ID: %s'%(self.th_id))
        plt.show(block=True)
        return 0



    def write_stats_open(self, rank):
        """Write all chisq and trim after eliminating trim_chisq > chisq"""
        self.write_stats = {'dist' : open(self.specs_dist + '_{}.csv'.format(rank), 'w'),
                            'bad'  : open(self.specs_bad  + '_{}.csv'.format(rank), 'w')}
        self.write_stats['dist'].write('THING_ID, specs, accepted \n')
        self.write_stats['bad'].write('Spec with pure noise: Chisq<2 \n')
        self.write_stats['bad'].write('THING_ID, file, chisq \n')




    def write_stats_close(self):
        """Close files"""
        for i in ['dist', 'bad']: self.write_stats[i].close()




    def write_stats_file(self, name, *output):
        """Write dist files"""
        self.write_stats[name].write('\t'.join(map(str, output)) + '\n')
        self.write_stats[name].flush()




    def write_fits(self, result, lpix):
         """Write fits file for each heakpix pixel"""

         file_name = os.path.join(self.pix_dir, 'pix_%s.fits'%(lpix))
         fits = fitsio.FITS(file_name, 'rw', clobber=True)
         for (thid, data) in result:
            nrows   = len(data.index)
            data    = data.reset_index().to_dict(orient='list')
            names   = list(data)
            formats = ['f4']*len(names)
            fdata   = np.zeros(nrows, dtype=dict(names= names, formats=formats))
            fdata   = {key:np.array(value) for key,value in data.items()}

            fits.write(fdata, header={'Healpix':'%s'%(lpix),
                                      self.full_file:'Npix_side =%s'%(self.Npix_side)},
                              extname=str(thid))
         fits.close()
         if self.verbose: print ('... Writing FITS file: %s'%(lpix))




    def master_fits(self, all_info):
        """Write master file that contains info for all measurements accepted"""
        all_info = [j for vals in all_info for j in vals]
        lpix, thid, tmp = list(zip(*all_info))
        plate, mjd, fiberid, z_war, z_err, zshift = list(zip(*tmp))

        data = {}
        data['HEALPIX']   = np.array(lpix,            dtype='i4')
        data['THING_ID']  = np.array(thid,            dtype='i4')
        data['PLATE']     = np.array(plate,           dtype='i4')
        data['MJD']       = np.array(mjd,             dtype='i4')
        data['FIBERID']   = np.array(fiberid,         dtype='i4')
        data['ZWARNING']  = np.array(z_war,           dtype='i4')
        data['Z_ERR']     = np.array(z_err,           dtype='f4')
        data['Z']         = np.array(zshift,          dtype='f4')

        if self.verbose: print (' ... Writing MASTER-FITS file')
        os.system("rm master_file.fits")
        fits = fitsio.FITS('master_file.fits','rw')
        fits.write(data)
        fits.close()





    def plot_stats(self, size):
        """Collect all accepted Specs and plot a distribution"""

        print ('... stats are on specs_dist/')
        total_dist = []
        for i in np.arange(size):
            dist   = pd.read_csv(self.specs_dist + '_{}.csv'.format(i), sep='\s',
                      names = ['THING_ID', 'specs', 'accepted'], skiprows=1)
            total_dist.append(dist)
        df = pd.concat(total_dist)

        index  = np.arange(0,19)
        nspecs = np.arange(1,19)
        dict   = defaultdict(list)
        tot_spec  = df['specs'].sum()
        top_index = ['%2.f'%(df.query('specs == %s'%(i)).accepted.sum()) for i in nspecs]
        for i in nspecs:
            for j in index:
                dict[i].append(len(df.query('specs == %s and accepted == %s'%(i, j))) if j <= i else 0 )

        final  = pd.DataFrame(index=index, data=dict)
        labels = final.sort(ascending=False).as_matrix()

        plt.figure(figsize = (18, 9))
        ax   = plt.subplot(111)
        tmp  = sns.heatmap(final.sort(ascending=False).replace(0,-100), linewidths=0.5, annot=labels,  fmt="d",
                linecolor='white', cmap="YlGnBu",  vmax= 100, ax=ax)
        cbar = tmp.collections[0].colorbar
        cbar.set_label('# of THING_ID', rotation=270)
        cbar.set_ticks([100])
        cbar.set_ticklabels(["100%"])
        plt.ylabel('# Accepted specs per THING_ID')
        plt.xlabel('# Specs per THING_ID')
        plt.title('Total Specs : %s,    Unique THING_ID : %s'%(tot_spec, len(df)), y=1.08)
        plt.legend(loc = 'best')
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(nspecs - 1)
        ax2.set_xticklabels(top_index)
        plt.xlabel('Accepted Specs')
        plt.savefig('File_dist.pdf')
        plt.show(block=True)


    def plot_bad_spec(self, size):
        total_dist = []
        for i in np.arange(size):
            dist   = pd.read_csv(self.specs_bad + '_{}.csv'.format(i), sep='\s',
                      names = ['THING_ID', 'specs', 'chisq'], skiprows=2)
            total_dist.append(dist)
        df = pd.concat(total_dist)
        print([tuple(x) for x in df[['specs','chisq']].values])
        for names, chisq in [tuple(x) for x in df[['specs','chisq']].values]: #df['specs'].tolist():
            plate, name  = names.split('/')
            if not os.path.isfile(self.dir_spec + names): self.get_bnl_files(plate, names)
            df_file = read_fits(self.dir_spec , names, self.spec_cols).set_index('loglam')
            df_file['flux'].plot()
            print(name.replace('.fits',''))
            plt.savefig('/bad_specs/File_chisq_%.2f_%s.png'%(chisq, name.replace('.fits','')))
            plt.show(block=True)


if __name__=='__main__':
    print ("goofing around :P ")
    Qsos    = Qso_catalog(None, verbose = True)
    #Qsos.plot_stats(8)
    Qsos.plot_bad_spec(8)
