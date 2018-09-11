# Matias Carrasco Kind
#
# DESCUT
#

FROM ubuntu:18.04
MAINTAINER Matias Carrasco Kind <mgckind@gmail.com>

#ENVS
ENV HOME /root
ENV SHELL /bin/bash
ENV TERM xterm
ENV DEBIAN_FRONTEND noninteractive

# BASICS
RUN apt-get update -y && \
    apt-get install -y \
       git curl wget net-tools build-essential vim unzip libaio1 pkg-config \ 
       gfortran zlibc zlib1g zlib1g-dev tmux gcc libopenmpi-dev bzip2 \ 
       ca-certificates libhdf5-serial-dev hdf5-tools python3-dev openmpi-bin \
       openmpi-common binutils && \
    apt-get clean && apt-get purge && rm -rf  /var/lib/apt/lists/* /tmp/* /var/tmp/*


#CONDA
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

RUN conda install  easyaccess -c mgckind 
RUN conda install -y pip 

#PYTHON 3
RUN pip install --upgrade pip
RUN pip install cx_Oracle==5.3
RUN pip install requests
RUN pip install easyaccess==1.4.4 --no-deps --ignore-installed
RUN pip install  Pillow
RUN pip install  pyfits
RUN pip install astropy
RUN pip install mpi4py


ENV USER des
ENV HOME /home/des
RUN useradd --create-home --shell /bin/bash ${USER} --uid 1001

RUN rm -fr ${HOME}/.openmpi && mkdir -p ${HOME}/.openmpi
ADD default-mca-params.conf ${HOME}/.openmpi/mca-params.conf

WORKDIR ${HOME}
RUN mkdir ${HOME}/test
ADD test/* ${HOME}/test/.
RUN chown -R ${USER}:${USER} ${HOME}
USER ${USER}
ENV SHELL /bin/bash
ENV TERM xterm

CMD ["sleep", "3600"]
