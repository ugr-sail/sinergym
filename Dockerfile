# Base on nrel/energyplus from Nicholas Long but using 
# Ubuntu, Python 3.11 and BCVTB
ARG UBUNTU_VERSION=22.04
FROM ubuntu:${UBUNTU_VERSION}

# Arguments for EnergyPlus version (default values of version 9.5.0 if is not specified)
ARG ENERGYPLUS_VERSION=9.5.0
ARG ENERGYPLUS_INSTALL_VERSION=9-5-0
ARG ENERGYPLUS_SHA=de239b2e5f

# Argument for Sinergym extras libraries
ARG SINERGYM_EXTRAS=[extras]

# Argument for choosing Python version
ARG PYTHON_VERSION=3.11

ENV ENERGYPLUS_VERSION=$ENERGYPLUS_VERSION
ENV ENERGYPLUS_TAG=v$ENERGYPLUS_VERSION
ENV ENERGYPLUS_SHA=$ENERGYPLUS_SHA

# This should be x.y.z, but EnergyPlus convention is x-y-z
ENV ENERGYPLUS_INSTALL_VERSION=$ENERGYPLUS_INSTALL_VERSION
ENV EPLUS_PATH=/usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION

# Downloading from Github
# e.g. https://github.com/NREL/EnergyPlus/releases/download/v9.5.0/EnergyPlus-9.5.0-de239b2e5f-Linux-Ubuntu$UBUNTU_VERSION-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_BASE_URL https://github.com/NREL/EnergyPlus/releases/download/$ENERGYPLUS_TAG
ENV ENERGYPLUS_DOWNLOAD_FILENAME EnergyPlus-$ENERGYPLUS_VERSION-$ENERGYPLUS_SHA-Linux-Ubuntu20.04-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_URL $ENERGYPLUS_DOWNLOAD_BASE_URL/$ENERGYPLUS_DOWNLOAD_FILENAME
ENV BCVTB_PATH=/usr/local/bcvtb

# Collapse the update of packages, download and installation into one command
# to make the container smaller & remove a bunch of the auxiliary apps/files
# that are not needed in the container
RUN apt-get update && apt-get upgrade -y \
    # General apt modules requires
    && apt-get install -y ca-certificates curl libx11-6 libexpat1 git wget iputils-ping pandoc \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata \
    #Energyplus installation
    && curl -SLO $ENERGYPLUS_DOWNLOAD_URL \
    && chmod +x $ENERGYPLUS_DOWNLOAD_FILENAME \
    && echo "y\r" | ./$ENERGYPLUS_DOWNLOAD_FILENAME \
    && rm $ENERGYPLUS_DOWNLOAD_FILENAME \
    && cd /usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION \
    && rm -rf PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor PreProcess/IDFVersionUpdater \
    # Remove the broken symlinks
    && cd /usr/local/bin find -L . -type l -delete \
    # BCVTB installation
    && echo "Y\r" | apt-get install default-jre openjdk-8-jdk \ 
    && wget http://github.com/lbl-srg/bcvtb/releases/download/v1.6.0/bcvtb-install-linux64-v1.6.0.jar \
    && yes "1" | java -jar bcvtb-install-linux64-v1.6.0.jar \
    && cp -R 1/ $BCVTB_PATH && rm -R 1/ \
    # PYTHON
    && apt update -y \
    # Install python software
    && apt-get install -y software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    # Add ppa:deadsnakes/ppa in order to choose python version and not SO default
    && add-apt-repository ppa:deadsnakes/ppa \
    # Remove default SO python version
    && apt-get remove --auto-remove python3.10 -y \
    && apt update -y \
    # Install custom version
    && apt install python$PYTHON_VERSION -y \
    # Link version to python and python3 command
    && ln -s /usr/bin/python$PYTHON_VERSION /usr/bin/python3 \
    && ln -s /usr/bin/python$PYTHON_VERSION /usr/bin/python \
    # Download pip installer and execution (command line break custom python installation)
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \ 
    # clean files
    && apt-get autoremove -y && apt-get autoclean -y \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /sinergym
COPY requirements.txt .
COPY MANIFEST.in .
COPY setup.py .
COPY scripts /sinergym/scripts
COPY sinergym /sinergym/sinergym
COPY tests /sinergym/tests
COPY examples /sinergym/examples
RUN pip install -e .${SINERGYM_EXTRAS}

CMD ["/bin/bash"]

# Example Build: docker build -t sinergym:latest --build-arg ENERGYPLUS_VERSION=9.5.0 --build-arg ENERGYPLUS_INSTALL_VERSION=9-5-0 --build-arg ENERGYPLUS_SHA=de239b2e5f .
# Example Run: docker run -it --rm -p 5005:5005 sinergym:latest