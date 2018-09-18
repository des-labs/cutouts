# cutouts

Current working version: bulkthumbs_7.py

Current development version: #8

## Requirements (also in requirments.txt in master branch):
```
astropy == 3.0.3
easyaccess == 1.4.3
mpi4py == 2.0.0
mpich2 == 1.4.1p1 (should be included when getting mpi4py)
numpy == 1.13.3
pandas == 0.21.0 (should work with up-to-date 0.23.4)
pillow == 5.2.0 (this is the fork of PIL, *not* PIL itself)
```
## Options when running (from the built-in help):
```
usage: bulkthumbs_7.py [-h] [--csv CSV] [--ra [RA [RA ...]]]
                       [--dec [DEC [DEC ...]]] [--coadd [COADD [COADD ...]]]
                       [--make_tiffs] [--make_fits] [--make_pngs]
                       [--xsize XSIZE] [--ysize YSIZE] [--colors COLORS]
                       [--db DB]

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
  --xsize XSIZE         Size in arcminutes of the cutout x-axis. Default: 1.0
  --ysize YSIZE         Size in arcminutes of the cutout y-axis. Default: 1.0
  --colors COLORS       Color bands for the fits cutout. Default: i
  --db DB               Which database to use. Default: Y3A2 Options: DR1,
                        Y3A2.
```
### Example using a CSV:
`mpirun -n 6 python bulkthumbs_7.py --csv <user/path/to/csv/file> --make_pngs --make_fits --xsize 3 --ysize 2 --colors r,Y --db Y3A2`

### Example using a list of coordinates:
`mpirun -n 6 python bulkthumbs_7.py --ra 4.9703 10.1415 --dec -48.9987 -62.0329 --make_pngs --make_fits --xsize 1 --ysize 1 --colors g,r,i,z,Y --db Y3A2`
