# Base on nrel/energyplus from Nicholas Long but using 
# Ubuntu, Python 3.6 and BCVTB
FROM ubuntu:18.04

# Arguments for EnergyPlus version
ARG ENERGYPLUS_VERSION
ARG ENERGYPLUS_INSTALL_VERSION
ARG ENERGYPLUS_SHA

ENV ENERGYPLUS_VERSION=$ENERGYPLUS_VERSION
ENV ENERGYPLUS_TAG=v$ENERGYPLUS_VERSION
ENV ENERGYPLUS_SHA=$ENERGYPLUS_SHA

# This should be x.y.z, but EnergyPlus convention is x-y-z
ENV ENERGYPLUS_INSTALL_VERSION=$ENERGYPLUS_INSTALL_VERSION
ENV EPLUS_PATH=/usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION

# Downloading from Github
# e.g. https://github.com/NREL/EnergyPlus/releases/download/v8.3.0/EnergyPlus-8.3.0-6d97d074ea-Linux-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_BASE_URL https://github.com/NREL/EnergyPlus/releases/download/$ENERGYPLUS_TAG
ENV ENERGYPLUS_DOWNLOAD_FILENAME EnergyPlus-$ENERGYPLUS_VERSION-$ENERGYPLUS_SHA-Linux-x86_64.sh
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
    && rm -rf DataSets Documentation ExampleFiles WeatherData MacroDataSets PostProcess/convertESOMTRpgm \
    PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor PreProcess/IDFVersionUpdater

# Remove the broken symlinks
RUN cd /usr/local/bin find -L . -type l -delete

# Install Python
RUN apt-get update && echo "Y\r" | apt-get install python3.6 && echo "Y\r" | apt-get install python3-pip

# Install OpenJDK-8
RUN apt-get update && echo "Y\r" | apt-get install default-jre openjdk-8-jdk

# Install BCVTB
ENV BCVTB_PATH=/usr/local/bcvtb
RUN apt-get install wget \
    && wget http://github.com/lbl-srg/bcvtb/releases/download/v1.6.0/bcvtb-install-linux64-v1.6.0.jar
#    && echo -e "1\n\r\n$BCVTB_PATH\n" | java -jar bcvtb-install-linux64-v1.6.0.jar

# Working directory and copy files
WORKDIR /code
COPY requirements.txt .
COPY setup.py .
ADD energym /code/energym
RUN pip3 install -e .

CMD ["/bin/bash"]

#TODO Install BCVTB - Only the java -jar file.jar part, because of stupid prompted commands
#TODO Get ENERGYPLUS_INSTALL_VERSION from ENERGYPLUS_VERSION

# Build: docker build -t energyplus:8.6.0 --build-arg ENERGYPLUS_VERSION=8.6.0 --build-arg ENERGYPLUS_INSTALL_VERSION=8-6-0 --build-arg ENERGYPLUS_SHA=198c6a3cff .
# Run: docker run -it --rm -p 5005:5005 energyplus:8.6.0
