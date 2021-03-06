#!/usr/bin/env python

"""
TO ADD:
+ argparse csv filepath - don't require ra/dec or coadds in command line; open csv file in run() command
+ does y3a2 have tile info?
+ add log file
+ add alpha/delta-win to ra/dec query
+ add the fall-back fits files in MakeTiffCut
"""
#select * from (select coadd_object_id, alphawin_j2000, deltawin_j2000, ra, dec, tilename from dr1_main where tilename='DES0210-1624' and ra between 32.3011 and 32.8842 and dec between -16.685 and -16.1015) where rownum<101;

"""
Options: --make_tiff --make_fits --xsize 1 --ysize 1 --colors g,r,i,z,y --usernm lgelman2
des0210-1624_10_sample_no_tile_borders.csv
	It took 0.2696189880371094 seconds to sort the desired objects by tile.
	It took 1533836295.3904793 seconds to make the cutouts.
	It took 24.193700790405273 seconds for this program to run.
des0210-1624_100_sample_no_tile_borders.csv
	It took 1.30196213722229 seconds to sort the desired objects by tile.
	It took 1533841050.8522475 seconds to make the cutouts.
	It took 53.67267441749573 seconds for this program to run.
des0210-1624_100_sample.csv
	It took 1.2866945266723633 seconds to sort the desired objects by tile.
	It took 1533841136.8113585 seconds to make the cutouts.
	It took 53.73037242889404 seconds for this program to run.
des0210-1624_1000_sample.csv
	It took 14.975631713867188 seconds to sort the desired objects by tile.
	It took 1533842116.141917 seconds to make the cutouts.
	It took 354.807977437973 seconds for this program to run.
des0210-1624_10000_sample.csv
	It took 151.89701771736145 seconds to sort the desired objects by tile.
	It took 1533845648.2547967 seconds to make the cutouts.
	It took 3408.4696927070618 seconds for this program to run.
"""

import os, sys
import argparse
import fnmatch, re
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
from PIL import Image

Image.MAX_IMAGE_PIXELS = 144000000		# allows Pillow to not freak out at a large filesize
ARCMIN_TO_DEG = 0.0166667		# deg per arcmin

# Check if this object already has a record in our archive, then check if the cutout needs to be cropped to desired size
#def exists():
# Sort the user's inputted coordinate list by tile
#def sortList(uu, pp):

# Make the cutout file if necessary (see exists() above)
def MakeTiffCut(tiledir, outdir, im, positions, xs, ys, df, maketiff, makepngs):
	# try opening I band FITS (fallback on G, R bands)
		tilename = glob.glob(tiledir+'*_i.fits.fz')
		os.makedirs(outdir, exist_ok=True)
		try:
			hdul = fits.open(tilename[0])
		except IOError as e:
			print('Urgh!')
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
	# Iterate over al desired colors
	os.makedirs(outdir, exist_ok=True)
	for c in range(len(colors)):
		if colors[c] == 'Y':
			tilename = glob.glob(tiledir + '*_{}.fits.fz'.format(colors[c]))
		else:
			tilename = glob.glob(tiledir + '*_{}.fits.fz'.format(colors[c].lower()))
		try:
			hdul = fits.open(tilename[0])
		except IOError as e:
			print('No FITS file in {0} color band found. Will not create cutouts in this band.'.format(colors[c]))
			continue
		
		# Iterate over all inputted coordinates
		for p in range(len(positions)):
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
	start = time.time()
	
	usernm = args.usernm
	xs = float(args.xsize)
	ys = float(args.ysize)
	colors = args.colors.split(',')
	jobid = str(uuid.uuid4())
	outdir = args.usernm + '/' + jobid + '/'
	maketiff = args.make_tiff
	makepngs = args.make_pngs
	makefits = args.make_fits
	
	if args.db == 'DR1':
		db = 'desdr'
	elif args.db == 'Y3A2':
		db = 'dessci'
	
	userdf = pd.DataFrame(pd.read_csv('des0210-1624_100_sample.csv'))
	#userdf = pd.DataFrame(pd.read_csv('des0210-1624_100_sample_no_tile_borders.csv'))
	userdf.drop(columns=['ALPHAWIN_J2000', 'DELTAWIN_J2000', 'RA', 'DEC', 'TILENAME'], inplace=True)
	
	df = pd.DataFrame()
	unmatched_coords = {'RA':[], 'DEC':[]}
	unmatched_ids = []
	
	conn = ea.connect(db)
	curs = conn.cursor()
	
	if args.ra:
		for i in range(len(userdf)):
			ra = userdf['RA'][i]
			ra180 = ra
			if ra > 180:
				ra180 = 360 - ra
			if args.db == 'DR1':
				query = "select * from (select T.TILENAME, M.ALPAHWIN_J2000, M.DELTAWIN_J2000, M.RA, M.DEC from DR1_TILE_INFO T, DR1_MAIN M where M.TILENAME = T.TILENAME and (CROSSRA0='N' and ({0} between RACMIN and RACMAX) and ({1} between DECCMIN and DECCMAX)) or (CROSSRA0='Y' and ({2} between RACMIN-360 and RACMAX) and ({1} between DECCMIN and DECCMAX))) where rownum=1".format(ra, userdf['DEC'][i], ra180)
			#elif args.db.upper == 'Y3A2':
			#	
			f = conn.query_to_pandas(query)
			if f.empty:
				unmatched['RA'].append(userdf['RA'][i])
				unmatched['DEC'].append(userdf['DEC'][i])
			else:	
				df = df.append(f)
	#if args.coadd:
	if 'COADD_OBJECT_ID' in userdf:
		for i in range(len(userdf)):
			if args.db == 'DR1':
				query = "select COADD_OBJECT_ID, ALPHAWIN_J2000, DELTAWIN_J2000, RA, DEC, TILENAME from DR1_MAIN where COADD_OBJECT_ID={0}".format(userdf['COADD_OBJECT_ID'][i])
			#elif args.db.upper == 'Y3A2':
			#
			f = conn.query_to_pandas(query)
			if f.empty:
				unmatched_ids.append(userdf['COADD_OBJECT_ID'][i])
			else:
				df = df.append(f)
	
	conn.close()
	df = df.merge(userdf, on='COADD_OBJECT_ID', how='inner')
	#df = df.sort_values(by=['TILENAME'])			# 2018-08-09 - Isn't necessary when slicing dataframe by tilename below.
	
	dt1 = time.time() - start
	print('It took {} seconds to sort the desired objects by tile.'.format(dt1))
	
	size = u.Quantity((ys, xs), u.arcmin)
	#positions = SkyCoord(df['ALPHAWIN_J2000'], df['DELTAWIN_J2000'], frame='icrs', unit='deg', equinox='J2000', representation_type='spherical')
	#positions = SkyCoord([32.301142,32.301148], [-16.475058,-16.306904], frame='icrs', unit='deg', equinox='J2000', representation_type='spherical')
	#positions = SkyCoord([32.339727, 32.579913], [-16.486495, -16.402778], frame='icrs', unit='deg', equinox='J2000', representation_type='spherical')
	
	tilenm = df['TILENAME'].unique()
	for i in tilenm:
		tiledir = i + '/'
		udf = df[df.TILENAME == i]
		
		positions = SkyCoord(udf['ALPHAWIN_J2000'], udf['DELTAWIN_J2000'], frame='icrs', unit='deg', equinox='J2000', representation_type='spherical')
		
		if maketiff or makepngs:
			imgname = glob.glob(tiledir+'*.tiff')
			try:
				im = Image.open(imgname[0])
			except IOError as e:
				print('No TIFF file found. Will not create true-color cutout.')
			else:
				MakeTiffCut(tiledir, outdir, im, positions, xs, ys, udf, maketiff, makepngs)
		
		if args.make_fits:
			MakeFitsCut(tiledir, outdir, size, positions, colors, udf)
	
	dt2 = time.time() - dt1
	print('It took {} seconds to make the cutouts.'.format(dt2))
	
	end = dt2 - start
	print('It took {} seconds for this program to run.'.format(end))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="This program will make any number of cutouts, using the master tiles.")
	# user directory - for cutouts to go to when they are made
	parser.add_argument('--ra', nargs='*', required=False, help='RA (decimal degrees)')
	parser.add_argument('--dec', nargs='*', required=False, help='DEC (decimal degrees)')
	parser.add_argument('--coadd', nargs='*', required=False, help='Coadd ID for exact object matching.')
	parser.add_argument('--make_tiff', action='store_true', help='Creates a TIFF file of the cutout region.')
	parser.add_argument('--make_fits', action='store_true', help='Creates FITS files in the desired bands of the cutout region.')
	parser.add_argument('--make_pngs', action='store_true', help='Creates a PNG file of the cutout region.')
	parser.add_argument('--xsize', default=1.0, help='Size in arcminutes of the cutout x-axis. Default: 1.0')
	parser.add_argument('--ysize', default=1.0, help='Size in arcminutes of the cutout y-axis. Default: 1.0')
	parser.add_argument('--colors', default='I', type=str.upper, help='Color bands for the fits cutout. Default: i')
	parser.add_argument('--db', default='DR1', type=str.upper, required=False, help='Which database to use. Default: DR1 Options: DR1, Y3A2.')
	parser.add_argument('--usernm', required=False, help='Username for database; otherwise uses values from desservices file.')
	#parser.add_argument('--passwd', required=False, help='Password for databse; otherwise uses values from desservices file.')
	# output directory
	
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	
	args = parser.parse_args()
	
	#if not (ra and dec) and not coadd:
	#	print('Please include either RA/DEC coordinates or Coadd IDs.')
	#	sys.exit(1)
	#if (ra and dec) and len(ra) != len(dec):
	#	print('Remember to have the same number of RA and DEC values when using coordinates.')
	#	sys.exit(1)
	#if (ra and not dec) or (dec and not ra):
	#	print('Please include BOTH RA and DEC if not using Coadd IDs.')
	#	sys.exit(1)
	if not args.make_tiff and not args.make_pngs and not args.make_fits:
		print('Nothing to do. Please select either/both make_tiff and make_fits.')
		sys.exit(1)
	
	run(args)
