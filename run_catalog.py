
"""
 using mpi4py

** TODO:
    - self.THING_ID, flux, ivar
    - add comments, and del sdss_catalog
    - README has some typos, and difficult to read
"""


import math
import numpy as np
from qso_catalog import Qso_catalog
from main_file import split_pixel
from get_files import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


dir_files = 'data/'
file_name = 'subset_spAll-v5_10_0.csv'
#file_name = 'spAll-v5_10_0.fits'
#file_name = 'DR14Q_v1_1.fits'

spall_cols  = ['RA','DEC','THING_ID','MJD','PLATE','FIBERID','BOSS_TARGET1','CLASS',
                'EBOSS_TARGET0','EBOSS_TARGET1','OBJTYPE','Z','Z_ERR','ZWARNING']

#spall_cols  = ['RA','DEC','THING_ID','MJD','PLATE','FIBERID','Z','Z_ERR','ZWARNING']

print (size, rank)

if rank == 0:
    df_fits = read_sub_fits(dir_files, file_name)
    #df_fits = read_fits(dir_files, file_name, spall_cols)
    Qsos    = Qso_catalog(df_fits, verbose = True)

    Qsos.rep_thid    = 1
    Qsos.write_master= True
    Qsos.write_ffits = True
    Qsos.show_plots  = False
    Qsos.write_names = False
    Qsos.write_hist  = True
    Qsos.need_files  = False

    #Qsos.own_filter()
    Qsos.filtering_qsos(condition= Qsos.condition)
    unique_pixels = Qsos.adding_pixel_column()
    #print (Qsos.df_qsos.query('PIX == 6219 & THING_ID == 77964771'))

    if Qsos.write_names: Qsos.write_file_names()
   
    lenpix = len(unique_pixels)
    nchunk = int(math.ceil(lenpix*1./size))
    chunks = [unique_pixels[i:i+ nchunk] for i in range(0, lenpix, nchunk)]
else:
    chunks = []
    Qsos   = None


Qsos = comm.bcast(Qsos, root=0)
if Qsos.write_hist: Qsos.write_stats_open(rank)

chunk_pix  = comm.scatter(chunks, root=0)
split_pixel(chunk_pix, Qsos)
comm.Barrier()

all_info  = comm.gather(Qsos.all_info, root=0)

if Qsos.write_hist: Qsos.write_stats_close()
if rank == 0:
    if Qsos.write_hist and Qsos.show_plots: Qsos.plot_stats(size)
    if Qsos.write_master: Qsos.master_fits(all_info)


