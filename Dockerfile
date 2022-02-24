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
    PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor PreProcess/IDFVersionUpdater

# Remove the broken symlinks
RUN cd /usr/local/bin find -L . -type l -delete

# Install Python
RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN apt install python${PYTHON_VERSION} python${PYTHON_VERSION}-distutils -y
RUN apt install python3-pip -y

# Install enchant for sinergym documentation
RUN apt-get update && echo "Y\r" | apt-get install enchant --fix-missing -y
# Install OpenJDK-8
RUN apt-get update && echo "Y\r" | apt-get install default-jre openjdk-8-jdk

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
COPY setup.py .
COPY DRL_battery.py .
COPY sinergym /sinergym/sinergym
COPY tests /sinergym/tests
COPY examples /sinergym/examples
COPY check_run_times.py .
RUN pip install -e .${SINERGYM_EXTRAS}

CMD ["/bin/bash"]

# Build: docker build -t sinergym:1.1.0 --build-arg ENERGYPLUS_VERSION=9.5.0 --build-arg ENERGYPLUS_INSTALL_VERSION=9-5-0 --build-arg ENERGYPLUS_SHA=de239b2e5f .
# Run: docker run -it --rm -p 5005:5005 sinergym:1.1.0
