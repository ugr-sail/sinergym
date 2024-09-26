# Container for Sinergym development
# ---------------------------------------------------------------------------- #
#                                  BASE IMAGE                                  #
# ---------------------------------------------------------------------------- #

ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION}

# ---------------------------------------------------------------------------- #
#                      CONTAINER ARGUMENTS AND ENV CONFIG                      #
# ---------------------------------------------------------------------------- #

# -------------------------------- ENERGYPLUS -------------------------------- #

# VERSION ARGUMENTS
ARG ENERGYPLUS_VERSION=24.1.0
ARG ENERGYPLUS_INSTALL_VERSION=24-1-0
ARG ENERGYPLUS_SHA=9d7789a3ac

#ENV CONFIGURATION
ENV ENERGYPLUS_TAG=v$ENERGYPLUS_VERSION
ENV EPLUS_PATH=/usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION
# Downloading from Github
# e.g. https://github.com/NREL/EnergyPlus/releases/download/v23.1.0/EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_BASE_URL https://github.com/NREL/EnergyPlus/releases/download/$ENERGYPLUS_TAG
ENV ENERGYPLUS_DOWNLOAD_FILENAME EnergyPlus-$ENERGYPLUS_VERSION-$ENERGYPLUS_SHA-Linux-Ubuntu22.04-x86_64.sh 
ENV ENERGYPLUS_DOWNLOAD_URL $ENERGYPLUS_DOWNLOAD_BASE_URL/$ENERGYPLUS_DOWNLOAD_FILENAME
# Python add pyenergyplus path in order to detect API package
ENV PYTHONPATH="/usr/local/EnergyPlus-${ENERGYPLUS_INSTALL_VERSION}"

# ---------------------------------- PYTHON ---------------------------------- #

# VERSION ARGUMENT
ARG PYTHON_VERSION=3.12

# ENV CONFIGURATION
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ---------------------------------- POETRY ---------------------------------- #

# ENV CONFIGURATION
ENV POETRY_NO_INTERACTION=1 
ENV POETRY_VIRTUALENVS_IN_PROJECT=0
ENV POETRY_VIRTUALENVS_CREATE=0
ENV POETRY_CACHE_DIR=/tmp/poetry_cache

# ------------------------- WANDB API KEY (IF EXISTS) ------------------------ #

ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

# LC_ALL for python locale error (https://bobbyhadz.com/blog/locale-error-unsupported-locale-setting-in-python)
ENV LC_ALL=C

# ---------------------------------------------------------------------------- #
#                        INSTALLATION AND CONFIGURATION                        #
# ---------------------------------------------------------------------------- #

# --------------------- APT UPDATE AND MANDATORY PACKAGES -------------------- #

RUN apt update && apt upgrade -y
RUN apt install -y ca-certificates build-essential curl libx11-6 libexpat1 git wget python3

# -------------------------- ENERGYPLUS INSTALLATION ------------------------- #

RUN curl -SLO $ENERGYPLUS_DOWNLOAD_URL \
    && chmod +x $ENERGYPLUS_DOWNLOAD_FILENAME \
    && echo "y\r" | ./$ENERGYPLUS_DOWNLOAD_FILENAME \
    && rm $ENERGYPLUS_DOWNLOAD_FILENAME \
    && cd /usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION \
    && rm -rf PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor PreProcess/IDFVersionUpdater \
    # Remove the broken symlinks
    && cd /usr/local/bin find -L . -type l -delete

# ------------------------ PYTHON AND PIP CONFIGURTION ----------------------- #

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py \
    && pip install --upgrade pip setuptools wheel \
    && ln -s /usr/bin/python3 /usr/bin/python \
    # Install some apt dependencies
    && echo "Y\r" | apt install python3-enchant -y \
    && echo "Y\r" | apt install pandoc -y 

# ---------------------------- POETRY CONFIGURTION --------------------------- #

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# -------------------------------- CLEAN FILES ------------------------------- #

RUN apt autoremove -y && apt autoclean -y \
    && rm -rf /var/lib/apt/lists/* 

# ---------------------------------------------------------------------------- #
#                            WORKDIR AND COPY FILES                            #
# ---------------------------------------------------------------------------- #

WORKDIR /workspaces/sinergym
# Only copy basic files for poetry installation
COPY pyproject.toml poetry.lock README.md LICENSE INSTALL.md CODE_OF_CONDUCT.md ./

# ---------------------------------------------------------------------------- #
#                    SINERGYM PACKAGE INSTALLATION (POETRY)                    #
# ---------------------------------------------------------------------------- #

RUN poetry install --no-interaction --with dev

# Execute the command
CMD ["/bin/bash"]

