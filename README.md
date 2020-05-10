# cutouts

Current stable version: bulkthumbs_15.py

Current development version: ...

## Minimum Requirements (also in requirments.txt):
```
astropy >= 3.0.3
easyaccess == 1.4.7
json
mpi4py >= 3.0.0
mpich2 == 3.3.2
numpy == 1.18.1
pandas >= 1.0.1
pillow >= 7.0.0 (this is the fork of PIL, *not* PIL itself)
python >= 3.6
pyyaml
```
## Options when running (from the built-in help):
```
usage: bulkthumbs_15.py [-h] [--config CONFIG] [--csv CSV]
                        [--ra [RA [RA ...]]] [--dec [DEC [DEC ...]]]
                        [--coadd [COADD [COADD ...]]] [--make_tiffs]
                        [--make_fits] [--make_pngs] [--make_rgb_lupton]
                        [--make_rgb_stiff] [--return_list] [--xsize XSIZE]
                        [--ysize YSIZE] [--colors_fits COLORS_FITS]
                        [--colors_rgb R,G,B] [--rgb_minimum RGB_MINIMUM]
                        [--rgb_stretch RGB_STRETCH] [--rgb_asinh RGB_ASINH]
                        [--db DB] [--jobid JOBID] [--usernm USERNM]
                        [--passwd PASSWD] [--outdir OUTDIR]

This program will make any number of cutouts, using the master tiles.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Optional file to list all these arguments in and pass
                        it along to bulkthumbs.
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
  --make_rgb_lupton     Creates 3-color images from the color bands you
                        select, from reddest to bluest. This method uses the
                        Lupton RGB combination method.
  --make_rgb_stiff      Creates 3-color images from the color bands you
                        select, from reddest to bluest. This method uses the
                        program STIFF to combine the images.
  --return_list         Saves list of inputted objects and their matched tiles
                        to user directory.
  --xsize XSIZE         Size in arcminutes of the cutout x-axis. Default: 1.0
  --ysize YSIZE         Size in arcminutes of the cutout y-axis. Default: 1.0
  --colors_fits COLORS_FITS
                        Color bands for the fits cutout. Default: i
  --colors_rgb R,G,B    Bands from which to combine the the RGB image, e.g.:
                        z,r,g. Call multiple times for multiple colors
                        combinations, e.g.: --colors_rgb z,r,g --colors_rgb
                        z,i,r.
  --rgb_minimum RGB_MINIMUM
                        The black point for the 3-color image. Default 1.0
  --rgb_stretch RGB_STRETCH
                        The linear stretch of the image. Default 50.0.
  --rgb_asinh RGB_ASINH
                        The asinh softening parameter. Default 10.0
  --db DB               Which database to use. Default: Y3A2, Options: DR1,
                        Y3A2.
  --jobid JOBID         Option to manually specify a jobid for this job.
  --usernm USERNM       Username for database; otherwise uses values from
                        desservices file.
  --passwd PASSWD       Password for database; otherwise uses values from
                        desservices file.
  --outdir OUTDIR       Overwrite for output directory.
```
### Example using a CSV:
`mpirun -n 6 python bulkthumbs_15.py --csv /path/to/csv/file --make_pngs --make_fits --colors_fits r,Y --xsize 3 --ysize 2 --db Y3A2`

### Example using a list of coordinates:
`mpirun -n 6 python bulkthumbs_15.py --ra 4.9703 10.1415 --dec -48.9987 -62.0329 --make_pngs --make_fits --colors_fits g,r,i,z,Y --make_rgb_lupton --make_rgb_stiff --colors_rgb i,r,g --colors_rgb z,r,g --xsize 1 --ysize 1 --db Y3A2`
