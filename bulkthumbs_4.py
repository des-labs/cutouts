#!/usr/bin/env python

"""
TO ADD:
+ add log file
+ add master library functionality 
	+ use sqlite
	+ check if cutout for object ID and cutout size or larger already exists
	+ if yes, open that instead and make cut if different size needed
	+ save cuts that do not previously exist in master library
	+ overwrite files for objects pre-existing at smaller sizes (with a limit?)
+ add in "fail-safe" code to deal with missing tile fits
+ add in queries for using Y3A2?
+ Figure out a way to auto-crop the black region on the tiffs/pngs if the image is smaller than requested size (on the edge of tile)

SUGGESTED COMMAND:
mpirun -n 4 python bulkthumbs_4.py --csv des0210-1624_100_coadds_sample.csv --make_tiffs --make_fits --xsize 3 --ysize 2 --colors g,r,i --usernm lgelman2

TESTS:
Options: time mpirun -n 1 python bulkthumbs_4.py --csv des0210-1624_100_sample.csv --make_tiff --make_fits --xsize 1 --ysize 1 --colors g,r,i,z,y
	1 core: real 0m31s, user 0m3s, sys 0m1s
	2 cores: 

Options: time mpirun -n 6 python bulkthumbs_4.py --csv des_tiles_sample_1200_coadds.csv --make_tiffs --make_fits --xsize 1 --ysize 1 --colors g,r,i,z,y
1200 objects across 12 tiles, 7,200 files created totaling ?
	2 cores: real 8m18s, user 12m2s, sys 0m45s
	4 cores: real 7m19s, user 14m52s, sys 1m55s
	6 cores: real 8m51s, user 22m1s, sys 5m42s

Options: time mpirun -n 6 python bulkthumbs_4.py --csv des_tiles_sample_12000_coadds.csv --make_tiffs --make_fits --xsize 1 --ysize 1 --colors g,r,i,z,y
12,000 objects across 12 tiles, 72,000 files created totalling ~37 GiB
	2 cores: real 44m50s, user 80m13s, sys1m55s
	4 cores: real 32m8s, user 102m, sys 4m14s
	6 cores: real 35m49s, user 161m33s, sys 23m47s

Options: time mpirun -n 6 python bulkthumbs_4.py --csv des_tiles_sample_12000_coadds.csv --make_pngs --xsize 1 --ysize 1
12,000 objects across 12 tiles (1000/tile), 12,000 files created totalling 1.5 GiB
	2 cores: 
	4 cores: 
	6 cores: (ncsa) 4m11s, query 170s; (home) 11m26s, query 645s

Options: time mpirun -n 6 python bulkthumbs_4.py --csv des_tiles_sample_135518_coadds.csv --make_pngs --xsize 1 --ysize 1
135,518 objects across 12 tiles, 135,518 file created totalling 17.4 GiB
	6 cores: (ncsa) 46m40s, query 2358s; (home) 135m29s, query 7761s
"""

import os, sys
import argparse
import glob
import time
import easyaccess as ea
import numpy as np
import pandas as pd
import uuid
from astropy import units as u
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata import NoOverlapError
from astropy.nddata import utils as ndu
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import utils
from mpi4py import MPI as mpi
from PIL import Image

Image.MAX_IMAGE_PIXELS = 144000000		# allows Pillow to not freak out at a large filesize
ARCMIN_TO_DEG = 0.0166667		# deg per arcmin

comm = mpi.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

def MakeTiffCut(tiledir, outdir, im, positions, xs, ys, df, maketiff, makepngs):
	# try opening I band FITS (fallback on G, R bands)
	
	#tilename = glob.glob(tiledir+'*_i.fits.fz')
	os.makedirs(outdir, exist_ok=True)
	hdul = None
	for _i in ['i','g','r','z','Y']:
		tilename = glob.glob(tiledir+'*_{}.fits.fz'.format(_i))
		try:
			hdul = fits.open(tilename[0])
		except IOError as e:
			hdul = None
			continue
		else:
			break
	if not hdul:
		print('Cannot find a master fits file for this tile.')
		return
	
	w = WCS(hdul['SCI'].header)
	
	pixelscale = utils.proj_plane_pixel_scales(w)
	
	dx = int(0.5 * xs * ARCMIN_TO_DEG / pixelscale[0])		# pixelscale is in degrees (CUNIT)
	dy = int(0.5 * ys * ARCMIN_TO_DEG / pixelscale[1])
	
	pixcoords = utils.skycoord_to_pixel(positions, w, origin=0, mode='wcs')
	
	for i in range(len(positions)):
		if 'COADD_OBJECT_ID' in df:
			filenm = outdir + str(df['COADD_OBJECT_ID'][i])
		else:
			filenm = outdir + 'x{0}y{1}'.format(df['RA'][i], df['DEC'][i])
		left = pixcoords[0][i] - dx
		upper = im.size[1] - pixcoords[1][i] - dy
		right = pixcoords[0][i] + dx
		lower = im.size[1] - pixcoords[1][i] + dy
		newimg = im.crop((left, upper, right, lower))
		
		if maketiff:
			newimg.save(filenm+'.tiff', format='TIFF')
		if makepngs:
			newimg.save(filenm+'.png', format='PNG')

def MakeFitsCut(tiledir, outdir, size, positions, colors, df):
	os.makedirs(outdir, exist_ok=True)			# Check if outdir exists
	
	for c in range(len(colors)):		# Iterate over al desired colors
		# Finish the tile's name and open the file. Camel-case check is required because Y band is always capitalized.
		if colors[c] == 'Y':
			tilename = glob.glob(tiledir + '*_{}.fits.fz'.format(colors[c]))
		else:
			tilename = glob.glob(tiledir + '*_{}.fits.fz'.format(colors[c].lower()))
		try:
			hdul = fits.open(tilename[0])
		except IOError as e:
			print('No FITS file in {0} color band found. Will not create cutouts in this band.'.format(colors[c]))
			continue		# Just go on to the next color in the list
		
		for p in range(len(positions)):			# Iterate over all inputted coordinates
			if 'COADD_OBJECT_ID' in df:
				filenm = outdir + '{0}_{1}.fits'.format(df['COADD_OBJECT_ID'][p], colors[c].lower())
			else:
				filenm = outdir + 'x{0}y{1}_{2}.fits'.format(df['RA'][p], df['DEC'][p], colors[c].lower())
			newhdul = fits.HDUList()
			
			# Iterate over all HDUs in the tile
			for i in range(len(hdul)):
				if hdul[i].name == 'PRIMARY':
					continue
				
				h = hdul[i].header
				data = hdul[i].data
				header = h.copy()
				w=WCS(header)
				
				cutout = Cutout2D(data, positions[p], size, wcs=w, mode='trim')
				crpix1, crpix2 = cutout.position_cutout
				x, y = cutout.position_original
				crval1, crval2 = w.wcs_pix2world(x, y, 1)
				
				header['CRPIX1'] = crpix1
				header['CRPIX2'] = crpix2
				header['CRVAL1'] = str(crval1)
				header['CRVAL2'] = str(crval2)
				header['HIERARCH RA_CUTOUT'] = df['RA'][p]
				header['HIERARCH DEC_CUTOUT'] = df['DEC'][p]
				
				if not newhdul:
					newhdu = fits.PrimaryHDU(data=cutout.data, header=header)
				else:
					newhdu = fits.ImageHDU(data=cutout.data, header=header, name=h['EXTNAME'])
				newhdul.append(newhdu)
			
			newhdul.writeto(filenm, output_verify='exception', overwrite=True, checksum=False)
			newhdul.close()

def run(args):
	xs = float(args.xsize)
	ys = float(args.ysize)
	colors = args.colors.split(',')
	usernm = ''
	jobid = ''
	outdir = ''
	
	if rank == 0:
		start = time.time()
		if args.db == 'DR1':
			db = 'desdr'
		elif args.db == 'Y3A2':
			db = 'dessci'
		
		# This puts any input type into a pandas dataframe
		if args.csv:
			userdf = pd.DataFrame(pd.read_csv(args.csv))
		elif args.ra:
			coords = {}
			coords['RA'] = args.ra
			coords['DEC'] = args.dec
			userdf = pd.DataFrame.from_dict(coords, orient='columns')
		elif args.coadd:
			coadds = {}
			coadds['COADD_OBJECT_ID']
			userdf = pd.DataFrame.from_dict(coadds, orient='columns')
		
		df = pd.DataFrame()
		unmatched_coords = {'RA':[], 'DEC':[]}
		unmatched_coadds = []
		
		conn = ea.connect(db)
		curs = conn.cursor()
		
		usernm = str(conn.user)
		jobid = str(uuid.uuid4())
		outdir = usernm + '/' + jobid + '/'
		
		if 'RA' in userdf:
			print(userdf.head())
			
			for i in range(len(userdf)):
				ra = userdf['RA'][i]
				ra180 = ra
				if ra > 180:
					ra180 = 360 - ra
				
				if args.db == 'DR1':
					#query = "select * from (select T.TILENAME, M.ALPHAWIN_J2000, M.DELTAWIN_J2000, M.RA, M.DEC from DR1_TILE_INFO T, DR1_MAIN M where M.TILENAME = T.TILENAME and (CROSSRA0='N' and ({0} between RACMIN and RACMAX) and ({1} between DECCMIN and DECCMAX)) or (CROSSRA0='Y' and ({2} between RACMIN-360 and RACMAX) and ({1} between DECCMIN and DECCMAX))) where rownum=1".format(ra, userdf['DEC'][i], ra180)
					
					#query = "select * from (select T.CROSSRA0, M.TILENAME, M.ALPHAWIN_J2000, M.DELTAWIN_J2000, M.RA, M.DEC from DR1_TILE_INFO T, DR1_MAIN M where T.TILENAME = M.TILENAME and (T.CROSSRA0='N' and ({0} between T.RACMIN and T.RACMAX) and ({1} between T.DECCMIN and T.DECCMAX)) or (T.CROSSRA0='Y' and ({2} between T.RACMIN-360 and T.RACMAX) and ({1} between T.DECCMIN and T.DECCMAX))) where rownum=1".format(ra, userdf['DEC'][i], ra180)
					
					query = "select * from (select TILENAME from DR1_TILE_INFO where (CROSSRA0='N' and ({0} between RACMIN and RACMAX) and ({1} between DECCMIN and DECCMAX)) or (CROSSRA0='Y' and ({2} between RACMIN-360 and RACMAX) and ({1} between DECCMIN and DECCMAX))) where rownum=1".format(ra, userdf['DEC'][i], ra180)
					
				elif args.db.upper == 'Y3A2':
					query = "select * from (select TILENAME from Y3A2_COADDTILE_GEOM where (CROSSRA0='N' and ({0} between RACMIN and RACMAX) and ({1} between DECCMIN and DECCMAX)) or (CROSSRA0='Y' and ({2} between RACMIN-360 and RACMAX) and ({1} between DECCMIN and DECCMAX))) where rownum=1".format(ra, userdf['DEC'][i], ra180)
				
				f = conn.query_to_pandas(query)
				if f.empty:
					unmatched_coords['RA'].append(userdf['RA'][i])
					unmatched_coords['DEC'].append(userdf['DEC'][i])
				else:	
					df = df.append(f)
			print(df.head())
			df = df.join(userdf, how='right')
			print(df.head())
		
		if 'COADD_OBJECT_ID' in userdf:
			for i in range(len(userdf)):
				if args.db == 'DR1':
					query = "select COADD_OBJECT_ID, ALPHAWIN_J2000, DELTAWIN_J2000, RA, DEC, TILENAME from DR1_MAIN where COADD_OBJECT_ID={0}".format(userdf['COADD_OBJECT_ID'][i])
				elif args.db.upper == 'Y3A2':
					query = "select COADD_OBJECT_ID, ALPHAWIN_J2000, DELTAWIN_J2000, RA, DEC, TILENAME from Y3A2_COADD_OBJECT_SUMMARY where COADD_OBJECT_ID={0}".format(userdf['COADD_OBJECT_ID'][i])
				
				f = conn.query_to_pandas(query)
				if f.empty:
					unmatched_coadds.append(userdf['COADD_OBJECT_ID'][i])
				else:
					df = df.append(f)
		
		conn.close()
		print(df.head())
		df = df.sort_values(by=['TILENAME'])
		
		chunksize = int(df.shape[0] / nprocs) + (df.shape[0] % nprocs)
		df = [ df[ i:i+chunksize ] for i in range(0, df.shape[0], chunksize) ]
		
		end = time.time()
		print('Querying took (s): ' + str(end-start))
		print(unmatched_coords)
		print(unmatched_coadds)
	
	else:
		df = None
	
	usernm, jobid, outdir = comm.bcast([usernm, jobid, outdir], root=0)
	#outdir = usernm + '/' + jobid + '/'
	df = comm.scatter(df, root=0)
	
	tilenm = df['TILENAME'].unique()
	for i in tilenm:
		tiledir = 'tiles_sample/' + i + '/'
		udf = df[ df.TILENAME == i ]
		udf = udf.reset_index()
		
		size = u.Quantity((ys, xs), u.arcmin)
		positions = SkyCoord(udf['ALPHAWIN_J2000'], udf['DELTAWIN_J2000'], frame='icrs', unit='deg', equinox='J2000', representation_type='spherical')
		
		if args.make_tiffs or args.make_pngs:
			# 20180823 - I can't think of a good reason for this to NOT be included within the function above, so, merge?
			imgname = glob.glob(tiledir + '*.tiff')
			try:
				im = Image.open(imgname[0])
			except IOError as e:
				print('No TIFF file found for tile ' + str(i) + '. Will not create true-color cutout.')
			else:
				MakeTiffCut(tiledir, outdir, im, positions, xs, ys, udf, args.make_tiffs, args.make_pngs)
		
		if args.make_fits:
			MakeFitsCut(tiledir, outdir, size, positions, colors, udf)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="This program will make any number of cutouts, using the master tiles.")
	
	parser.add_argument('--csv', type=str, required=False, help='A CSV with columns \'COADD_OBJECT_ID \' or \'RA,DEC\'')
	parser.add_argument('--ra', nargs='*', required=False, type=float, help='RA (decimal degrees)')
	parser.add_argument('--dec', nargs='*', required=False, help='DEC (decimal degrees)')
	parser.add_argument('--coadd', nargs='*', required=False, help='Coadd ID for exact object matching.')
	
	parser.add_argument('--make_tiffs', action='store_true', help='Creates a TIFF file of the cutout region.')
	parser.add_argument('--make_fits', action='store_true', help='Creates FITS files in the desired bands of the cutout region.')
	parser.add_argument('--make_pngs', action='store_true', help='Creates a PNG file of the cutout region.')
	
	parser.add_argument('--xsize', default=1.0, help='Size in arcminutes of the cutout x-axis. Default: 1.0')
	parser.add_argument('--ysize', default=1.0, help='Size in arcminutes of the cutout y-axis. Default: 1.0')
	parser.add_argument('--colors', default='I', type=str.upper, help='Color bands for the fits cutout. Default: i')
	
	parser.add_argument('--db', default='DR1', type=str.upper, required=False, help='Which database to use. Default: DR1 Options: DR1, Y3A2.')
	#parser.add_argument('--usernm', required=False, help='Username for database; otherwise uses values from desservices file.')
	#parser.add_argument('--passwd', required=False, help='Password for database; otherwise uses values from desservices file.')
	
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	
	args = parser.parse_args()
	
	
	if not args.csv and not (args.ra and args.dec) and not args.coadd:
		print('Please include either RA/DEC coordinates or Coadd IDs.')
		sys.exit(1)
	if (args.ra and args.dec) and len(args.ra) != len(args.dec):
		print('Remember to have the same number of RA and DEC values when using coordinates.')
		sys.exit(1)
	if (args.ra and not args.dec) or (args.dec and not args.ra):
		print('Please include BOTH RA and DEC if not using Coadd IDs.')
		sys.exit(1)
	if not args.make_tiffs and not args.make_pngs and not args.make_fits:
		print('Nothing to do. Please select either/both make_tiff and make_fits.')
		sys.exit(1)
	
	run(args)
