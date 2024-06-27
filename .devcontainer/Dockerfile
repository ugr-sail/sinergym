# Base on nrel/energyplus from Nicholas Long but using 
# Ubuntu, Python 3.10 and BCVTB
ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION}

# Arguments for EnergyPlus version (default values of version 8.6.0 if is not specified)
ARG ENERGYPLUS_VERSION=23.1.0
ARG ENERGYPLUS_INSTALL_VERSION=23-1-0
ARG ENERGYPLUS_SHA=87ed9199d4

# Argument for Sinergym extras libraries
ARG SINERGYM_EXTRAS=[extras]

# Argument for choosing Python version
ARG PYTHON_VERSION=3.12
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# WANDB_API_KEY
ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

# LC_ALL for python locale error (https://bobbyhadz.com/blog/locale-error-unsupported-locale-setting-in-python)
ENV LC_ALL=C

ENV ENERGYPLUS_VERSION=$ENERGYPLUS_VERSION
ENV ENERGYPLUS_TAG=v$ENERGYPLUS_VERSION
ENV ENERGYPLUS_SHA=$ENERGYPLUS_SHA

# This should be x.y.z, but EnergyPlus convention is x-y-z
ENV ENERGYPLUS_INSTALL_VERSION=$ENERGYPLUS_INSTALL_VERSION
ENV EPLUS_PATH=/usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION

# Downloading from Github
# e.g. https://github.com/NREL/EnergyPlus/releases/download/v23.1.0/EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_BASE_URL https://github.com/NREL/EnergyPlus/releases/download/$ENERGYPLUS_TAG
ENV ENERGYPLUS_DOWNLOAD_FILENAME EnergyPlus-$ENERGYPLUS_VERSION-$ENERGYPLUS_SHA-Linux-Ubuntu22.04-x86_64.sh 
ENV ENERGYPLUS_DOWNLOAD_URL $ENERGYPLUS_DOWNLOAD_BASE_URL/$ENERGYPLUS_DOWNLOAD_FILENAME

# Mandatory apt packages
RUN apt update && apt upgrade -y
RUN apt install -y ca-certificates curl libx11-6 libexpat1 git wget

#Energyplus installation
RUN curl -SLO $ENERGYPLUS_DOWNLOAD_URL \
    && chmod +x $ENERGYPLUS_DOWNLOAD_FILENAME \
    && echo "y\r" | ./$ENERGYPLUS_DOWNLOAD_FILENAME \
    && rm $ENERGYPLUS_DOWNLOAD_FILENAME \
    && cd /usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION \
    && rm -rf PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor PreProcess/IDFVersionUpdater \
    # Remove the broken symlinks
    && cd /usr/local/bin find -L . -type l -delete

# Install pip, and make python point to python3
RUN apt install python3-pip -y \
    && ln -s /usr/bin/python3 /usr/bin/python \
    # Install some apt dependencies
    && echo "Y\r" | apt install python3-enchant -y \
    && echo "Y\r" | apt install pandoc -y 

# clean files
RUN apt autoremove -y && apt autoclean -y \
    && rm -rf /var/lib/apt/lists/* 
    

# Python add pyenergyplus path in order to detect API package
ENV PYTHONPATH="/usr/local/EnergyPlus-${ENERGYPLUS_INSTALL_VERSION}"

WORKDIR /workspaces/sinergym
COPY requirements.txt /workspaces/sinergym/requirements.txt
COPY MANIFEST.in /workspaces/sinergym/MANIFEST.in
COPY setup.py /workspaces/sinergym/setup.py
COPY scripts /workspaces/sinergym/scripts
COPY sinergym /workspaces/sinergym/sinergym
COPY tests /workspaces/sinergym/tests
COPY examples /workspaces/sinergym/examples
COPY docs/source /workspaces/sinergym/docs/source
COPY .git/ /workspaces/sinergym/.git
RUN pip install -e .${SINERGYM_EXTRAS}

#RUN pip install idna && pip install six
CMD ["/bin/bash"]

