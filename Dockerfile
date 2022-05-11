# Base on nrel/energyplus from Nicholas Long but using 
# Ubuntu, Python 3.6 and BCVTB
ARG UBUNTU_VERSION=18.04
FROM ubuntu:${UBUNTU_VERSION}

# Configuring tzdata in order to don't ask for geographic area
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Arguments for EnergyPlus version (default values of version 8.6.0 if is not specified)
ARG ENERGYPLUS_VERSION=9.5.0
ARG ENERGYPLUS_INSTALL_VERSION=9-5-0
ARG ENERGYPLUS_SHA=de239b2e5f

# Argument for Sinergym extras libraries
ARG SINERGYM_EXTRAS=[extras]

# Argument for choose Python version
ARG PYTHON_VERSION=3.9

ENV ENERGYPLUS_VERSION=$ENERGYPLUS_VERSION
ENV ENERGYPLUS_TAG=v$ENERGYPLUS_VERSION
ENV ENERGYPLUS_SHA=$ENERGYPLUS_SHA

# This should be x.y.z, but EnergyPlus convention is x-y-z
ENV ENERGYPLUS_INSTALL_VERSION=$ENERGYPLUS_INSTALL_VERSION
ENV EPLUS_PATH=/usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION

# Downloading from Github
# e.g. https://github.com/NREL/EnergyPlus/releases/download/v9.5.0/EnergyPlus-9.5.0-de239b2e5f-Linux-Ubuntu18.04-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_BASE_URL https://github.com/NREL/EnergyPlus/releases/download/$ENERGYPLUS_TAG
ENV ENERGYPLUS_DOWNLOAD_FILENAME EnergyPlus-$ENERGYPLUS_VERSION-$ENERGYPLUS_SHA-Linux-Ubuntu18.04-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_URL $ENERGYPLUS_DOWNLOAD_BASE_URL/$ENERGYPLUS_DOWNLOAD_FILENAME

# Collapse the update of packages, download and installation into one command
# to make the container smaller & remove a bunch of the auxiliary apps/files
# that are not needed in the container
RUN apt-get update \
    && apt-get install -y ca-certificates curl libx11-6 libexpat1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -SLO $ENERGYPLUS_DOWNLOAD_URL \
    && chmod +x $ENERGYPLUS_DOWNLOAD_FILENAME \
    && echo "y\r" | ./$ENERGYPLUS_DOWNLOAD_FILENAME \
    && rm $ENERGYPLUS_DOWNLOAD_FILENAME \
    && cd /usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION \
    && rm -rf PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor PreProcess/IDFVersionUpdater

# Remove the broken symlinks
RUN cd /usr/local/bin find -L . -type l -delete

# Install ping dependency
RUN apt update && apt install iputils-ping -y 

# Install Python version PYTHON_VERSION
RUN apt update \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install python${PYTHON_VERSION} python${PYTHON_VERSION}-distutils -y \
    && apt install python3-pip -y \
    && ln -s /usr/bin/pip3 /usr/bin/pip \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Install enchant for sinergym documentation
RUN apt-get update && echo "Y\r" | apt-get install enchant --fix-missing -y
# Install OpenJDK-8
RUN apt-get update && echo "Y\r" | apt-get install default-jre openjdk-8-jdk

# START Install pandoc for sinergym jupyter documentation

# install tzdata, which we will need down the line
RUN apt-get install -f -y --no-install-recommends tzdata
RUN echo "Now installing cabal, and other tools."
# We install cabal, a Haskell package manager, because we want the newest
# pandoc and filters which we can only get from there.
# We also install zlib1g, as we will need it later on.
# We install librsvg2 in order to make svg -> pdf conversation possible.
# imagemagick may be needed by the latex-formulae-pandoc filter
RUN apt-get install -f -y --no-install-recommends \
      cabal-install \
      librsvg2-bin \
      librsvg2-common \
      zlib1g \
      zlib1g-dev

# get the newest list of packages
RUN echo "Getting the newest list of cabal packages."
RUN cabal update
# update cabal
RUN cabal install cabal-install

RUN echo "Installing QuickCheck via cabal."
RUN cabal install --ghc-options='+RTS -M2G -RTS' \
                  QuickCheck
RUN echo "Installing pandoc via cabal."
RUN cabal install --ghc-options='+RTS -M2G -RTS' \
                  --allow-newer=base \
                  pandoc
RUN echo "Installing pandoc-citeproc via cabal."
RUN cabal install --ghc-options='+RTS -M2G -RTS' \
                  --allow-newer=base \
                  pandoc-citeproc
RUN echo "Installing pandoc-citeproc-preamble via cabal."
RUN cabal install --ghc-options='+RTS -M2G -RTS' \
                  --allow-newer=base \
                  pandoc-citeproc-preamble
RUN echo "Installing pandoc-crossref via cabal."
RUN cabal install --ghc-options='+RTS -M2G -RTS' \
                  pandoc-crossref
# clear unnecessary cabal files
RUN echo "Performing cleanup of unnecessary cabal files."
RUN rm -rf /root/.cabal/logs &&\
    rm -rf /root/.cabal/packages &&\
    rm -rf /root/.cabal/lib &&\
    rm -rf /root/.cabal/share/doc &&\
    rm -rf /root/.cabal/share/man &&\
    (find /root/.cabal/ -type f -empty -delete || true) &&\
    (find /root/.cabal/ -type d -empty -delete || true)
# remove cabal-install and dependencies
RUN echo "Removing cabal-install and its dependencies. They are no longer needed."
RUN apt-get purge -f -y cabal-install ghc &&\
    apt-get autoremove -f -y &&\
    apt-get autoclean -y &&\
    apt-get autoremove -f -y &&\
    apt-get autoclean -y
# clean up all temporary files
RUN echo "Performing more cleanup."
RUN apt-get clean -y &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -f /etc/ssh/ssh_host_*
RUN echo "Done."

# we remember the path to pandoc in a special variable
ENV PANDOC_DIR=/root/.cabal/bin/

# add pandoc to the path
ENV PATH=${PATH}:${PANDOC_DIR}

# End pandoc installation



# Install BCVTB
ENV BCVTB_PATH=/usr/local/bcvtb
RUN apt-get install wget \
    && wget http://github.com/lbl-srg/bcvtb/releases/download/v1.6.0/bcvtb-install-linux64-v1.6.0.jar \
    && yes "1" | java -jar bcvtb-install-linux64-v1.6.0.jar \
    && cp -R 1/ $BCVTB_PATH && rm -R 1/

# Working directory and copy files
RUN python -m pip install --upgrade pip
# Upgrade setuptools for possible errors (depending on python version)
RUN pip install --upgrade setuptools
RUN apt-get update && apt-get upgrade -y && apt-get install -y git

WORKDIR /sinergym
COPY requirements.txt .
COPY MANIFEST.in .
COPY setup.py .
COPY DRL_battery.py .
COPY load_agent.py .
COPY sinergym /sinergym/sinergym
COPY tests /sinergym/tests
COPY examples /sinergym/examples
COPY check_run_times.py .
COPY try_env.py .
RUN pip install -e .${SINERGYM_EXTRAS}


#uninstall 3.6 python default version
RUN apt-get remove --purge python3-pip python3 -y \
    && apt-get autoremove -y && apt-get autoclean -y
RUN pip install idna
CMD ["/bin/bash"]

# Build: docker build -t sinergym:1.1.0 --build-arg ENERGYPLUS_VERSION=9.5.0 --build-arg ENERGYPLUS_INSTALL_VERSION=9-5-0 --build-arg ENERGYPLUS_SHA=de239b2e5f .
# Run: docker run -it --rm -p 5005:5005 sinergym:1.1.0
