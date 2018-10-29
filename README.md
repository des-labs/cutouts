# cutouts

Current stable version: bulkthumbs_12.py

Current development version: ...

## Requirements (also in requirments.txt in master branch):
```
astropy == 3.0.3
easyaccess == 1.4.3
json
mpi4py == 2.0.0
mpich2 == 1.4.1p1 (should be included when getting mpi4py)
numpy == 1.13.3
pandas == 0.21.0 (should work with up-to-date 0.23.4)
pillow == 5.2.0 (this is the fork of PIL, *not* PIL itself)
yaml
```
## Options when running (from the built-in help):
```
usage: bulkthumbs_12.py [-h] [--csv CSV] [--ra [RA [RA ...]]]
                        [--dec [DEC [DEC ...]]] [--coadd [COADD [COADD ...]]]
                        [--make_tiffs] [--make_fits] [--make_pngs]
                        [--make_rgbs MAKE_RGBS] [--return_list]
                        [--xsize XSIZE] [--ysize YSIZE] [--colors COLORS]
                        [--rgb_minimum RGB_MINIMUM]
                        [--rgb_stretch RGB_STRETCH] [--rgb_asinh RGB_ASINH]
                        [--db DB] [--jobid JOBID]

This program will make any number of cutouts, using the master tiles.

optional arguments:
  -h, --help            show this help message and exit
  --csv CSV             A CSV with columns 'COADD_OBJECT_ID ' or 'RA,DEC'
  --ra [RA [RA ...]]    RA (decimal degrees)
  --dec [DEC [DEC ...]]
                        DEC (decimal degrees)
  --coadd [COADD [COADD ...]]
                        Coadd ID for exact object matching.
  --make_tiffs          Creates a TIFF file of the cutout region.
  --make_fits           Creates FITS files in the desired bands of the cutout
                        region.
  --make_pngs           Creates a PNG file of the cutout region.
  --make_rgbs           Creates 3-color images using the bands you select
                        (reddest to bluest), e.g.: --make_rgbs i,r,g
                        --make_rgbs z,i,r --make_rgbs z,r,g
  --return_list         Saves list of inputted objects and their matched tiles
                        to user directory.
  --xsize XSIZE         Size in arcminutes of the cutout x-axis. Default: 1.0
  --ysize YSIZE         Size in arcminutes of the cutout y-axis. Default: 1.0
  --colors              Color bands for the fits cutout. Default: i
  --rgb_minimum         The black point for the 3-color image. Default 1.0
  --rgb_stretch         The linear stretch of the image. Default 50.0.
  --rgb_asinh           The asinh softening parameter. Default 10.0
  --db DB               Which database to use. Default: Y3A2, Options: DR1,
                        Y3A2.
  --jobid               Option to manually specify a jobid for this job.
```
### Example using a CSV:
`mpirun -n 6 python bulkthumbs_9.py --csv /path/to/csv/file --make_pngs --make_fits --colors r,Y --xsize 3 --ysize 2 --db Y3A2`

### Example using a list of coordinates:
`mpirun -n 6 python bulkthumbs_9.py --ra 4.9703 10.1415 --dec -48.9987 -62.0329 --make_pngs --make_fits --colors g,r,i,z,Y --make_rgbs i,r,g --make_rgb z,r,g --xsize 1 --ysize 1 --db DR1`
