#!/usr/bin/env python

"""
TESTS:
Options: time mpirun -n 6 python bulkthumbs_8.py --csv des_tiles_sample_135518_coadds.csv --make_pngs --xsize 1 --ysize 1
	6 cores: (ncsa) 

Options: time mpirun -n 6 python bulkthumbs_8.py --csv des_tiles_sample_129412_coords.csv --make_pngs --xsize 1 --ysize 1
	6 cores: (ncsa) 
"""

import os, sys
import argparse
import datetime
import logging
import glob
import time
import easyaccess as ea
import numpy as np
import pandas as pd
import uuid
import json
import yaml
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

TILES_FOLDER = '' #'tiles/'
OUTDIR = '' #'output/'

comm = mpi.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

class MPILogHandler(logging.FileHandler):
	def __init__(self, filename, comm, amode=mpi.MODE_WRONLY|mpi.MODE_CREATE|mpi.MODE_APPEND):
		self.comm = comm
		self.filename = filename
		self.amode = amode
		self.encoding = 'utf-8'
		logging.StreamHandler.__init__(self, self._open())
	def _open(self):
		stream = mpi.File.Open(self.comm, self.filename, self.amode)
		stream.Set_atomicity(True)
		return stream
	def emit(self, record):
		try:
			msg = self.format(record)
			stream = self.stream
			stream.Write_shared((msg+self.terminator).encode(self.encoding))
		except Exception:
			self.handleError(record)
	def close(self):
		if self.stream:
			self.stream.Sync()
			self.stream.Close()
			self.stream = None

def getPathSize(path):
	dirsize = 0
	for entry in os.scandir(path):
		if entry.is_dir(follow_symlinks=False):
			dirsize += getPathSize(entry.path)
		else:
			dirsize += os.path.getsize(entry)
	return dirsize

def _DecConverter(ra, dec):
	ra1 = np.abs(ra/15)
	raHH = int(ra1)
	raMM = int((ra1 - raHH) * 60)
	raSS = (((ra1 - raHH) * 60) - raMM) * 60
	raSS = np.round(raSS, decimals=1)
	raOUT = '{0:02d}{1:02d}{2:04.1f}'.format(raHH, raMM, raSS) if ra > 0 else '-{0:02d}{1:02d}{2:04.1f}'.format(raHH, raMM, raSS)
	
	dec1 = np.abs(dec)
	decDD = int(dec1)
	decMM = int((dec1 - decDD) * 60)
	decSS = (((dec1 - decDD) * 60) - decMM) * 60
	decSS = np.round(decSS, decimals=1)
	decOUT = '-{0:02d}{1:02d}{2:04.1f}'.format(decDD, decMM, decSS) if dec < 0 else '+{0:02d}{1:02d}{2:04.1f}'.format(decDD, decMM, decSS)
	
	return raOUT + decOUT

def MakeTiffCut(tiledir, outdir, positions, xs, ys, df, maketiff, makepngs):
	logger = logging.getLogger(__name__)
	os.makedirs(outdir, exist_ok=True)
	
	imgname = glob.glob(tiledir + '*.tiff')
	try:
		im = Image.open(imgname[0])
	except IndexError as e:
		print('No TIFF file found for tile ' + df['TILENAME'][0] + '. Will not create true-color cutout.')
		logger.error('MakeTiffCut - No TIFF file found for tile ' + df['TILENAME'][0] + '. Will not create true-color cutout.')
		return
	
	# try opening I band FITS (fallback on G, R bands)
	hdul = None
	for _i in ['i','g','r','z','Y']:
		tilename = glob.glob(tiledir+'*_{}.fits.fz'.format(_i))
		try:
			hdul = fits.open(tilename[0])
		except IOError as e:
			hdul = None
			logger.warning('MakeTiffCut - Could not find master FITS file: ' + tilename)
			continue
		else:
			break
	if not hdul:
		print('Cannot find a master fits file for this tile.')
		logger.error('MakeTiffCut - Cannot find a master fits file for this tile.')
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
			#filenm = outdir + 'DESJ' + _DecConverter(df['RA'][0], df['DEC'][0])
		left = max(0, pixcoords[0][i] - dx)
		upper = max(0, im.size[1] - pixcoords[1][i] - dy)
		right = min(pixcoords[0][i] + dx, 10000)
		lower = min(im.size[1] - pixcoords[1][i] + dy, 10000)
		newimg = im.crop((left, upper, right, lower))
		
		if maketiff:
			filenm += '.tiff'
			newimg.save(filenm, format='TIFF')
		if makepngs:
			filenm += '.png'
			newimg.save(filenm, format='PNG')
		if newimg.size != (2*dx, 2*dy):
			logger.info('MakeTiffCut - {} is smaller than user requested. This is likely because the object/coordinate was in close proximity to the edge of a tile.'.format(('/').join(filenm.split('/')[-2:])))
	logger.info('MakeTiffCut - Tile {} complete.'.format(df['TILENAME'][0]))

def MakeFitsCut(tiledir, outdir, size, positions, colors, df):
	logger = logging.getLogger(__name__)
	os.makedirs(outdir, exist_ok=True)			# Check if outdir exists
	
	for c in range(len(colors)):		# Iterate over al desired colors
		# Finish the tile's name and open the file. Camel-case check is required because Y band is always capitalized.
		if colors[c] == 'Y':
			tilename = glob.glob(tiledir + '*_{}.fits.fz'.format(colors[c]))
		else:
			tilename = glob.glob(tiledir + '*_{}.fits.fz'.format(colors[c].lower()))
		try:
			hdul = fits.open(tilename[0])
		except IndexError as e:
			print('No FITS file in {0} color band found. Will not create cutouts in this band.'.format(colors[c]))
			logger.error('MakeFitsCut - No FITS file in {0} color band found. Will not create cutouts in this band.'.format(colors[c]))
			continue		# Just go on to the next color in the list
		
		for p in range(len(positions)):			# Iterate over all inputted coordinates
			if 'COADD_OBJECT_ID' in df:
				filenm = outdir + '{0}_{1}.fits'.format(df['COADD_OBJECT_ID'][p], colors[c].lower())
			else:
				filenm = outdir + 'x{0}y{1}_{2}.fits'.format(df['RA'][p], df['DEC'][p], colors[c].lower())
				#filenm = outdir + 'DESJ' + _DecConverter(df['RA'][p], df['DEC'][p]) + '_{}.fits'.format(colors[c].lower())
			
			newhdul = fits.HDUList()
			pixelscale = None
			
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
					pixelscale = utils.proj_plane_pixel_scales(w)
				else:
					newhdu = fits.ImageHDU(data=cutout.data, header=header, name=h['EXTNAME'])
				newhdul.append(newhdu)
			
			if pixelscale is not None:
				dx = int(size[1] * ARCMIN_TO_DEG / pixelscale[0] / u.arcmin)		# pixelscale is in degrees (CUNIT)
				dy = int(size[0] * ARCMIN_TO_DEG / pixelscale[1] / u.arcmin)
				if (newhdul[0].header['NAXIS1'], newhdul[0].header['NAXIS2']) != (dx, dy):
					logger.info('MakeFitsCut - {} is smaller than user requested. This is likely because the object/coordinate was in close proximity to the edge of a tile.'.format(('/').join(filenm.split('/')[-2:])))
			
			newhdul.writeto(filenm, output_verify='exception', overwrite=True, checksum=False)
			newhdul.close()
	logger.info('MakeFitsCut - Tile {} complete.'.format(df['TILENAME'][0]))

def run(args):
	logtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
	logname = OUTDIR + 'BulkThumbs_' + logtime + '.log'
	formatter = logging.Formatter('%(asctime)s - '+str(rank)+' - %(levelname)-8s - %(message)s')
	
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	
	fh = MPILogHandler(logname, comm)
	fh.setLevel(logging.INFO)
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	
	xs = float(args.xsize)
	ys = float(args.ysize)
	colors = args.colors.split(',')
	usernm = ''
	jobid = ''
	outdir = ''
	
	if rank == 0:
		#sumlog = open(OUTDIR + 'BulkThumbs_'+logtime+'_SUMMARY.log', 'w')
		summary = {}
		start = time.time()
		if args.db == 'DR1':
			db = 'desdr'
		elif args.db == 'Y3A2':
			db = 'dessci'
		
		logger.info('Selected Options:')
		#sumlog.write('Selected Options: \n')
		
		# This puts any input type into a pandas dataframe
		if args.csv:
			userdf = pd.DataFrame(pd.read_csv(args.csv))
			logger.info('    CSV: '+args.csv)
			#sumlog.write('    CSV: ' + args.csv + '\n')
			summary['csv'] = args.csv
		elif args.ra:
			coords = {}
			coords['RA'] = args.ra
			coords['DEC'] = args.dec
			userdf = pd.DataFrame.from_dict(coords, orient='columns')
			logger.info('    RA: '+str(args.ra))
			logger.info('    DEC: '+str(args.dec))
			#sumlog.write('    RA: '+str(args.ra)+'\n')
			#sumlog.write('    DEC: '+str(args.dec)+'\n')
			summary['ra'] = str(args.ra)
			summary['dec'] = str(args.dec)
		elif args.coadd:
			coadds = {}
			coadds['COADD_OBJECT_ID'] = args.coadd
			userdf = pd.DataFrame.from_dict(coadds, orient='columns')
			logger.info('    CoaddID: '+str(args.coadd))
			#sumlog.write('    CoaddID: '+str(args.coadd)+'\n')
			summary['coadd'] = str(args.coadd)
		
		logger.info('    X size: '+str(args.xsize))
		logger.info('    Y size: '+str(args.ysize))
		logger.info('    Make TIFFs? '+str(args.make_tiffs))
		logger.info('    Make PNGs? '+str(args.make_pngs))
		logger.info('    Make FITS? '+str(args.make_fits))
		#sumlog.write('    X size: '+str(args.xsize)+'\n')
		#sumlog.write('    Y size: '+str(args.ysize)+'\n')
		#sumlog.write('    Make TIFFs? '+str(args.make_tiffs)+'\n')
		#sumlog.write('    Make PNGs? '+str(args.make_pngs)+'\n')
		#sumlog.write('    Make FITS? '+str(args.make_fits)+'\n')
		summary['xsize'] = str(args.xsize)
		summary['ysize'] = str(args.ysize)
		summary['make_tiffs'] = str(args.make_tiffs)
		summary['make_pngs'] = str(args.make_pngs)
		summary['make_fits'] = str(args.make_fits)
		if args.make_fits:
			logger.info('        Bands: '+args.colors)
			#sumlog.write('        Bands: '+args.colors)+'\n'
			summary['bands'] = args.colors
		summary['db'] = args.db
		
		df = pd.DataFrame()
		unmatched_coords = {'RA':[], 'DEC':[]}
		unmatched_coadds = []
		
		logger.info('Connecting to: '+db)
		conn = ea.connect(db)
		curs = conn.cursor()
		
		usernm = str(conn.user)
		jobid = str(uuid.uuid4())
		outdir = OUTDIR + usernm + '/' + jobid + '/'
		
		logger.info('User: ' + usernm)
		logger.info('JobID: ' + str(jobid))
		summary['user'] = usernm
		summary['jobid'] = str(jobid)
		
		tablename = 'BTL_'+jobid.upper().replace("-","_")	# "BulkThumbs_List_<jobid>"
		if 'RA' in userdf:
			if args.db == 'Y3A2':
				ra_adjust = [360-userdf['RA'][i] if userdf['RA'][i]>180 else userdf['RA'][i] for i in range(len(userdf['RA']))]
				userdf = userdf.assign(RA_ADJUSTED = ra_adjust)
				userdf.to_csv(OUTDIR+tablename+'.csv', index=False)
				conn.load_table(OUTDIR+tablename+'.csv', name=tablename)
				
				query = "select temp.RA, temp.DEC, temp.RA_ADJUSTED, temp.RA as ALPHAWIN_J2000, temp.DEC as DELTAWIN_J2000, m.TILENAME from {} temp left outer join Y3A2_COADDTILE_GEOM m on (m.CROSSRA0='N' and (temp.RA between m.URAMIN and m.URAMAX) and (temp.DEC between m.UDECMIN and m.UDECMAX)) or (m.CROSSRA0='Y' and (temp.RA_ADJUSTED between m.URAMIN-360 and m.URAMAX) and (temp.DEC between m.UDECMIN and m.UDECMAX))".format(tablename)
				
				df = conn.query_to_pandas(query)
				curs.execute('drop table {}'.format(tablename))
				#os.remove(OUTDIR+tablename+'.csv')
				
				df = df.replace('-9999',np.nan)
				dftemp = df[df.isnull().any(axis=1)]
				unmatched_coords['RA'] = dftemp['RA'].tolist()
				unmatched_coords['DEC'] = dftemp['DEC'].tolist()
			
			if args.db == 'DR1':
				for i in range(len(userdf)):
					ra = userdf['RA'][i]
					ra180 = ra
					if ra > 180:
						ra180 = 360 - ra
					
					if args.db == 'DR1':
						query = "select * from (select TILENAME from DR1_TILE_INFO where (CROSSRA0='N' and ({0} between RACMIN and RACMAX) and ({1} between DECCMIN and DECCMAX)) or (CROSSRA0='Y' and ({2} between RACMIN-360 and RACMAX) and ({1} between DECCMIN and DECCMAX))) where rownum=1".format(ra, userdf['DEC'][i], ra180)
					
					f = conn.query_to_pandas(query)
					if f.empty:
						unmatched_coords['RA'].append(userdf['RA'][i])
						unmatched_coords['DEC'].append(userdf['DEC'][i])
					else:	
						df = df.append(f)
			logger.info('Unmatched coordinates: \n{0}\n{1}'.format(unmatched_coords['RA'], unmatched_coords['DEC']))
			summary['Unmatched_Coords'] = unmatched_coords
			print(unmatched_coords)
		
		if 'COADD_OBJECT_ID' in userdf:
			if args.db == 'Y3A2':
				userdf.to_csv(OUTDIR+tablename+'.csv', index=False)
				conn.load_table(OUTDIR+tablename+'.csv', name=tablename)
				
				query = "select temp.COADD_OBJECT_ID, m.ALPHAWIN_J2000, m.DELTAWIN_J2000, m.RA, m.DEC, m.TILENAME from {} temp left outer join Y3A2_COADD_OBJECT_SUMMARY m on temp.COADD_OBJECT_ID=m.COADD_OBJECT_ID".format(tablename)
				
				df = conn.query_to_pandas(query)
				curs.execute('drop table {}'.format(tablename))
				os.remove(OUTDIR+tablename+'.csv')
				
				df = df.replace('-9999',np.nan)
				df = df.replace(-9999.000000,np.nan)
				dftemp = df[df.isnull().any(axis=1)]
				unmatched_coadds = dftemp['COADD_OBJECT_ID'].tolist()
			
			if args.db == 'DR1':
				for i in range(len(userdf)):
					query = "select COADD_OBJECT_ID, ALPHAWIN_J2000, DELTAWIN_J2000, RA, DEC, TILENAME from DR1_MAIN where COADD_OBJECT_ID={0}".format(userdf['COADD_OBJECT_ID'][i])
					
					f = conn.query_to_pandas(query)
					if f.empty:
						unmatched_coadds.append(userdf['COADD_OBJECT_ID'][i])
					else:
						df = df.append(f)
			logger.info('Unmatched coadd ID\'s: \n{}'.format(unmatched_coadds))
			summary['Unmatched_Coadds'] = unmatched_coadds
			print(unmatched_coadds)
		
		conn.close()
		df = df.sort_values(by=['TILENAME'])
		df = np.array_split(df, nprocs)
		
		end1 = time.time()
		query_elapsed = '{0:.2f}'.format(end1-start)
		print('Querying took (s): ' + query_elapsed)
		logger.info('Querying took (s): ' + query_elapsed)
		summary['query_time'] = query_elapsed
	
	else:
		df = None
	
	usernm, jobid, outdir = comm.bcast([usernm, jobid, outdir], root=0)
	#outdir = usernm + '/' + jobid + '/'
	df = comm.scatter(df, root=0)
	
	tilenm = df['TILENAME'].unique()
	for i in tilenm:
		tiledir = TILES_FOLDER + i + '/'
		#tiledir = 'DES0210-1624/'
		udf = df[ df.TILENAME == i ]
		udf = udf.reset_index()
		
		size = u.Quantity((ys, xs), u.arcmin)
		positions = SkyCoord(udf['ALPHAWIN_J2000'], udf['DELTAWIN_J2000'], frame='icrs', unit='deg', equinox='J2000', representation_type='spherical')
		
		if args.make_tiffs or args.make_pngs:
			MakeTiffCut(tiledir, outdir+i+'/', positions, xs, ys, udf, args.make_tiffs, args.make_pngs)
		
		if args.make_fits:
			MakeFitsCut(tiledir, outdir+i+'/', size, positions, colors, udf)
	
	comm.Barrier()
	
	if rank == 0:
		end2 = time.time()
		processing_time = '{0:.2f}'.format(end2-end1)
		print('Processing took (s): ' + processing_time)
		logger.info('Processing took (s): ' + processing_time)
		summary['processing_time'] = processing_time
		
		#pt1 = time.time()
		dirsize = getPathSize(outdir)
		dirsize = dirsize * 1. / 1024
		if dirsize > 1024. * 1024:
			dirsize = '{0:.2f} GB'.format(1. * dirsize / 1024. / 1024)
		elif dirsize > 1024.:
			dirsize = '{0:.2f} MB'.format(1. * dirsize / 1024.)
		else:
			dirsize = '{0:.2f} KB'.format(dirsize)
		
		logger.info('All processes finished.')
		logger.info('Total file size on disk: {}'.format(dirsize))
		summary['size_on_disk'] = str(dirsize)
		#pt2 = time.time()
		#print('{} seconds'.format(pt2 - pt1))
		
		pt3 = time.time()
		files = glob.glob(outdir + '*/*')
		logger.info('Total number of files: {}'.format(len(files)))
		summary['number_of_files'] = len(files)
		files = [i.split('/')[-2:] for i in files]
		files = [('/').join(i) for i in files]
		files = [i.split('.')[-2] for i in files]
		files = [i.split('_')[0] for i in files]
		files = list(set(files))
		summary['files'] = files
		pt4 = time.time()
		print('File adjusting time: '.format(pt4-pt3))
		
		jsonfile = OUTDIR + 'BulkThumbs_'+logtime+'_SUMMARY.json'
		with open(jsonfile, 'w') as fp:
			json.dump(summary, fp)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="This program will make any number of cutouts, using the master tiles.")
	
	parser.add_argument('--csv', type=str, required=False, help='A CSV with columns \'COADD_OBJECT_ID \' or \'RA,DEC\'')
	parser.add_argument('--ra', nargs='*', required=False, type=float, help='RA (decimal degrees)')
	parser.add_argument('--dec', nargs='*', required=False, type=float, help='DEC (decimal degrees)')
	parser.add_argument('--coadd', nargs='*', required=False, help='Coadd ID for exact object matching.')
	
	parser.add_argument('--make_tiffs', action='store_true', help='Creates a TIFF file of the cutout region.')
	parser.add_argument('--make_fits', action='store_true', help='Creates FITS files in the desired bands of the cutout region.')
	parser.add_argument('--make_pngs', action='store_true', help='Creates a PNG file of the cutout region.')
	
	parser.add_argument('--xsize', default=1.0, help='Size in arcminutes of the cutout x-axis. Default: 1.0')
	parser.add_argument('--ysize', default=1.0, help='Size in arcminutes of the cutout y-axis. Default: 1.0')
	parser.add_argument('--colors', default='I', type=str.upper, help='Color bands for the fits cutout. Default: i')
	
	parser.add_argument('--db', default='Y3A2', type=str.upper, required=False, help='Which database to use. Default: Y3A2 Options: DR1 (very slow), Y3A2 (much faster).')
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
	
	with open('config/bulkthumbsconfig.yaml','r') as cfile:
		conf = yaml.load(cfile)
	TILES_FOLDER = conf['directories']['tiles'] + '/'
	OUTDIR = conf['directories']['outdir'] + '/'
	
	run(args)
