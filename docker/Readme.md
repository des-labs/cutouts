# copy file
cp ../bulkthumbs_5.py bulkthumbs.py
# build image
docker build -t cutouts .
# run container
docker run -it -v /home/matias/DES/dr1_tiles:/tiles -v /home/matias/Dropbox/GitHub/des_labs_org/cutouts/docker/output:/output cutouts bash
# inside container run commands
