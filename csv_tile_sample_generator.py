#!/usr/bin/env python

import easyaccess as ea
import pandas as pd
import random

m = 10000
total = 0
r = random.randint(1, m)
total += r
query1 = 'select * from (select t.RA, t.DEC from DR1_MAIN t, DR1_TILE_INFO m where t.TILENAME=m.TILENAME and (t.RA between m.URAMIN and m.URAMAX) and (t.DEC between m.UDECMIN and m.UDECMAX) and t.TILENAME=\'DES0120-2706\') where rownum<{};'.format(r)
r = random.randint(1, m)
total += r
query2 = 'select * from (select t.RA, t.DEC from DR1_MAIN t, DR1_TILE_INFO m where t.TILENAME=m.TILENAME and (t.RA between m.URAMIN and m.URAMAX) and (t.DEC between m.UDECMIN and m.UDECMAX) and t.TILENAME=\'DES0133-4914\') where rownum<{};'.format(r)
r = random.randint(1, m)
total += r
query3 = 'select * from (select t.RA, t.DEC from DR1_MAIN t, DR1_TILE_INFO m where t.TILENAME=m.TILENAME and (t.RA between m.URAMIN and m.URAMAX) and (t.DEC between m.UDECMIN and m.UDECMAX) and t.TILENAME=\'DES0148-1707\') where rownum<{};'.format(r)
r = random.randint(1, m)
total += r
query4 = 'select * from (select t.RA, t.DEC from DR1_MAIN t, DR1_TILE_INFO m where t.TILENAME=m.TILENAME and (t.RA between m.URAMIN and m.URAMAX) and (t.DEC between m.UDECMIN and m.UDECMAX) and t.TILENAME=\'DES0202-3749\') where rownum<{};'.format(r)
r = random.randint(1, m)
total += r
query5 = 'select * from (select t.RA, t.DEC from DR1_MAIN t, DR1_TILE_INFO m where t.TILENAME=m.TILENAME and (t.RA between m.URAMIN and m.URAMAX) and (t.DEC between m.UDECMIN and m.UDECMAX) and t.TILENAME=\'DES0210-1624\') where rownum<{};'.format(r)
r = random.randint(1, m)
total += r
query6 = 'select * from (select t.RA, t.DEC from DR1_MAIN t, DR1_TILE_INFO m where t.TILENAME=m.TILENAME and (t.RA between m.URAMIN and m.URAMAX) and (t.DEC between m.UDECMIN and m.UDECMAX) and t.TILENAME=\'DES0227-0958\') where rownum<{};'.format(r)
r = random.randint(1, m)
total += r
query7 = 'select * from (select t.RA, t.DEC from DR1_MAIN t, DR1_TILE_INFO m where t.TILENAME=m.TILENAME and (t.RA between m.URAMIN and m.URAMAX) and (t.DEC between m.UDECMIN and m.UDECMAX) and t.TILENAME=\'DES0232-3206\') where rownum<{};'.format(r)
r = random.randint(1, m)
total += r
query8 = 'select * from (select t.RA, t.DEC from DR1_MAIN t, DR1_TILE_INFO m where t.TILENAME=m.TILENAME and (t.RA between m.URAMIN and m.URAMAX) and (t.DEC between m.UDECMIN and m.UDECMAX) and t.TILENAME=\'DES0308-1958\') where rownum<{};'.format(r)
r = random.randint(1, m)
total += r
query9 = 'select * from (select t.RA, t.DEC from DR1_MAIN t, DR1_TILE_INFO m where t.TILENAME=m.TILENAME and (t.RA between m.URAMIN and m.URAMAX) and (t.DEC between m.UDECMIN and m.UDECMAX) and t.TILENAME=\'DES0311-3749\') where rownum<{};'.format(r)
r = random.randint(1, m)
total += r
query10 = 'select * from (select t.RA, t.DEC from DR1_MAIN t, DR1_TILE_INFO m where t.TILENAME=m.TILENAME and (t.RA between m.URAMIN and m.URAMAX) and (t.DEC between m.UDECMIN and m.UDECMAX) and t.TILENAME=\'DES0435-2623\') where rownum<{};'.format(r)
r = random.randint(1, m)
total += r
query11 = 'select * from (select t.RA, t.DEC from DR1_MAIN t, DR1_TILE_INFO m where t.TILENAME=m.TILENAME and (t.RA between m.URAMIN and m.URAMAX) and (t.DEC between m.UDECMIN and m.UDECMAX) and t.TILENAME=\'DES2147-5540\') where rownum<{};'.format(r)
r = random.randint(1, m)
total += r
query12 = 'select * from (select t.RA, t.DEC from DR1_MAIN t, DR1_TILE_INFO m where t.TILENAME=m.TILENAME and (t.RA between m.URAMIN and m.URAMAX) and (t.DEC between m.UDECMIN and m.UDECMAX) and t.TILENAME=\'DES2246-4914\') where rownum<{};'.format(r)

"""
total = 0
r = random.randint(1, 20000)
total += r
query1 = 'select * from (select COADD_OBJECT_ID from DR1_MAIN where TILENAME=\'DES0120-2706\') where rownum<{};'.format(r)
r = random.randint(1, 20000)
total += r
query2 = 'select * from (select COADD_OBJECT_ID from DR1_MAIN where TILENAME=\'DES0133-4914\') where rownum<{};'.format(r)
r = random.randint(1, 20000)
total += r
query3 = 'select * from (select COADD_OBJECT_ID from DR1_MAIN where TILENAME=\'DES0148-1707\') where rownum<{};'.format(r)
r = random.randint(1, 20000)
total += r
query4 = 'select * from (select COADD_OBJECT_ID from DR1_MAIN where TILENAME=\'DES0202-3749\') where rownum<{};'.format(r)
r = random.randint(1, 20000)
total += r
query5 = 'select * from (select COADD_OBJECT_ID from DR1_MAIN where TILENAME=\'DES0210-1624\') where rownum<{};'.format(r)
r = random.randint(1, 20000)
total += r
query6 = 'select * from (select COADD_OBJECT_ID from DR1_MAIN where TILENAME=\'DES0227-0958\') where rownum<{};'.format(r)
r = random.randint(1, 20000)
total += r
query7 = 'select * from (select COADD_OBJECT_ID from DR1_MAIN where TILENAME=\'DES0232-3206\') where rownum<{};'.format(r)
r = random.randint(1, 20000)
total += r
query8 = 'select * from (select COADD_OBJECT_ID from DR1_MAIN where TILENAME=\'DES0308-1958\') where rownum<{};'.format(r)
r = random.randint(1, 20000)
total += r
query9 = 'select * from (select COADD_OBJECT_ID from DR1_MAIN where TILENAME=\'DES0311-3749\') where rownum<{};'.format(r)
r = random.randint(1, 20000)
total += r
query10 = 'select * from (select COADD_OBJECT_ID from DR1_MAIN where TILENAME=\'DES0435-2623\') where rownum<{};'.format(r)
r = random.randint(1, 20000)
total += r
query11 = 'select * from (select COADD_OBJECT_ID from DR1_MAIN where TILENAME=\'DES2147-5540\') where rownum<{};'.format(r)
r = random.randint(1, 20000)
total += r
query12 = 'select * from (select COADD_OBJECT_ID from DR1_MAIN where TILENAME=\'DES2246-4914\') where rownum<{};'.format(r)
"""

conn = ea.connect('desdr')
curs = conn.cursor()

df1 = conn.query_to_pandas(query1)
df2 = conn.query_to_pandas(query2)
df3 = conn.query_to_pandas(query3)
df4 = conn.query_to_pandas(query4)
df5 = conn.query_to_pandas(query5)
df6 = conn.query_to_pandas(query6)
df7 = conn.query_to_pandas(query7)
df8 = conn.query_to_pandas(query8)
df9 = conn.query_to_pandas(query9)
df10 = conn.query_to_pandas(query10)
df11 = conn.query_to_pandas(query11)
df12 = conn.query_to_pandas(query12)

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12], ignore_index=True)
df = df.round(4)
df.to_csv('des_tiles_sample_{}_coords.csv'.format(total), index=False)
#df.to_csv('des_tiles_sample_{}_coadds.csv'.format(total), index=False)
