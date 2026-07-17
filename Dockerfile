# ============================================================================ #
#                                  BASE IMAGE                                  #
# ============================================================================ #
# Build arguments shared across stages
ARG UBUNTU_VERSION=24.04

# ---------------------------------------------------------------------------- #
#                                  BUILDER STAGE                               #
# ---------------------------------------------------------------------------- #
# Installs everything: system deps, EnergyPlus, Poetry and Python packages.
# Acts as the common ancestor for the `dev` stage and as the source of
# artifacts copied into the `runtime` stage.
FROM --platform=linux/amd64 ubuntu:${UBUNTU_VERSION} AS builder

# ---------------------------------------------------------------------------- #
#                      CONTAINER ARGUMENTS AND ENV CONFIG                      #
# ---------------------------------------------------------------------------- #

# -------------------------------- ENERGYPLUS -------------------------------- #

# VERSION ARGUMENTS
ARG ENERGYPLUS_VERSION=25.1.0
ARG ENERGYPLUS_INSTALL_VERSION=25-1-0
ARG ENERGYPLUS_SHA=1c11a3d85f

# ENV CONFIGURATION
ENV ENERGYPLUS_TAG=v$ENERGYPLUS_VERSION
ENV EPLUS_PATH=/usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION
# Downloading from Github
# e.g. https://github.com/NREL/EnergyPlus/releases/download/v23.1.0/EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_BASE_URL=https://github.com/NREL/EnergyPlus/releases/download/$ENERGYPLUS_TAG-WithDSOASpaceListFixes
ENV ENERGYPLUS_DOWNLOAD_FILENAME=EnergyPlus-$ENERGYPLUS_VERSION-$ENERGYPLUS_SHA-Linux-Ubuntu24.04-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_URL=$ENERGYPLUS_DOWNLOAD_BASE_URL/$ENERGYPLUS_DOWNLOAD_FILENAME
# Python add pyenergyplus path in order to detect API package
ENV PYTHONPATH="/usr/local/EnergyPlus-${ENERGYPLUS_INSTALL_VERSION}"

# ---------------------------------- POETRY ---------------------------------- #

# Pin Poetry to >=2.4,<3.0: get patch releases automatically but guard against
# major-version breakage (e.g. lock file format changes). Bump the upper bound
# manually when ready to migrate to Poetry 3.x.
ARG POETRY_VERSION=">=2.4,<3.0"

# ENV CONFIGURATION
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=0
ENV POETRY_VIRTUALENVS_CREATE=0
ENV POETRY_CACHE_DIR=/tmp/poetry_cache

# ------------------------------- VIRTUAL ENV -------------------------------- #

ENV VENV_PATH=/opt/venv
ENV VIRTUAL_ENV="$VENV_PATH"
ENV PATH="$VENV_PATH/bin:$PATH"

# ------------------------- SINERGYM EXTRA LIBRARIES ------------------------- #

ARG SINERGYM_EXTRAS=""

# LC_ALL for python locale error (https://bobbyhadz.com/blog/locale-error-unsupported-locale-setting-in-python)
ENV LC_ALL=C

# ---------------------------------------------------------------------------- #
#                        INSTALLATION AND CONFIGURATION                        #
# ---------------------------------------------------------------------------- #

# --------------------- APT UPDATE AND MANDATORY PACKAGES -------------------- #

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        curl \
        libx11-6 \
        libexpat1 \
        git \
        wget \
        openssh-client \
        python3 \
        python3-venv \
        python3-enchant \
        pandoc \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# -------------------------- ENERGYPLUS INSTALLATION ------------------------- #

RUN curl -SLO "$ENERGYPLUS_DOWNLOAD_URL" \
    && chmod +x "$ENERGYPLUS_DOWNLOAD_FILENAME" \
    && printf 'y\n' | ./"$ENERGYPLUS_DOWNLOAD_FILENAME" \
    && rm "$ENERGYPLUS_DOWNLOAD_FILENAME" \
    && cd "/usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION" \
    && rm -rf PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor PreProcess/IDFVersionUpdater \
    && cd /usr/local/bin && find -L . -type l -delete

# ------------------------ PYTHON VENV AND POETRY ---------------------------- #

RUN python3 -m venv "$VENV_PATH" \
    && pip install --upgrade pip setuptools wheel \
    && pip install "poetry$POETRY_VERSION"

# ---------------------------------------------------------------------------- #
#                            WORKDIR AND COPY FILES                            #
# ---------------------------------------------------------------------------- #

WORKDIR /workspaces/sinergym

# Copy only dependency manifests first so that poetry install is cached
# when only the source code changes.
COPY pyproject.toml poetry.lock ./
COPY .coveragerc ./

# ---------------------------------------------------------------------------- #
#                    SINERGYM DEPENDENCY INSTALLATION (POETRY)                 #
# ---------------------------------------------------------------------------- #
# --no-root: install only dependencies, not the project itself (source not
# copied yet). The project is installed in editable mode in a second step
# after the source is available.

RUN poetry install --no-interaction --extras "$SINERGYM_EXTRAS" --no-root

# ----------------------------- COPY SOURCE CODE ----------------------------- #

COPY sinergym ./sinergym
COPY scripts ./scripts
COPY tests ./tests
COPY README.md INSTALL.md CODE_OF_CONDUCT.md LICENSE ./

# ---------------------------------------------------------------------------- #
#                    SINERGYM PROJECT INSTALLATION (POETRY)                    #
# ---------------------------------------------------------------------------- #
# Now that the source is present, install the project itself (editable mode).

RUN poetry install --no-interaction --extras "$SINERGYM_EXTRAS"

# -------------------------------- CLEAN CACHE ------------------------------- #

RUN rm -rf "$POETRY_CACHE_DIR"

# ---------------------------------------------------------------------------- #
#                                   DEV STAGE                                  #
# ---------------------------------------------------------------------------- #
# Full-featured image used by the devcontainer: keeps Poetry, build tools,
# test/doc/format dependencies and the full repository for development.
FROM builder AS dev

# Install development dependency groups (format, typing, test, doc, drl, gcloud,
# plots, notebooks) on top of the base dependencies already installed in the
# builder stage. The `dev` group already covers everything the extras would.
RUN poetry install --no-interaction --with dev

WORKDIR /workspaces/sinergym

CMD ["/bin/bash"]

# ---------------------------------------------------------------------------- #
#                                RUNTIME STAGE                                 #
# ---------------------------------------------------------------------------- #
# Minimal image for execution only. Copies EnergyPlus, the venv with installed
# Python packages and the scripts directory from the builder; no Poetry, no
# build tools, no test/doc dependencies.
FROM --platform=linux/amd64 ubuntu:${UBUNTU_VERSION} AS runtime

ARG ENERGYPLUS_INSTALL_VERSION=25-1-0

# Minimal runtime env — venv bin is on PATH so python/pytest/etc. resolve.
ENV PYTHONPATH="/usr/local/EnergyPlus-${ENERGYPLUS_INSTALL_VERSION}" \
    LC_ALL=C \
    EPLUS_PATH=/usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION \
    PATH="/usr/local/EnergyPlus-${ENERGYPLUS_INSTALL_VERSION}:/opt/venv/bin:${PATH}" \
    VENV_PATH=/opt/venv

# --------------------- MINIMAL APT PACKAGES FOR RUNTIME --------------------- #

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        libx11-6 \
        libexpat1 \
        libgomp1 \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ----------------------- COPY ARTIFACTS FROM BUILDER ----------------------- #

# EnergyPlus installation
COPY --from=builder /usr/local/EnergyPlus-${ENERGYPLUS_INSTALL_VERSION} /usr/local/EnergyPlus-${ENERGYPLUS_INSTALL_VERSION}
# Virtual environment with all Python packages installed by Poetry.
# Note: poetry installs the project in editable mode via a .pth pointing to
# /workspaces/sinergym, so the sinergym source package must be copied too.
COPY --from=builder /opt/venv /opt/venv
# sinergym package source (resolves the editable .pth) + data files
COPY --from=builder /workspaces/sinergym/sinergym /workspaces/sinergym/sinergym
# Scripts needed by the default CMD
COPY --from=builder /workspaces/sinergym/scripts /workspaces/sinergym/scripts
# Tests needed by CI (pytest tests/)
COPY --from=builder /workspaces/sinergym/tests /workspaces/sinergym/tests
# Coverage configuration needed by CI (pytest --cov sinergym)
COPY --from=builder /workspaces/sinergym/.coveragerc /workspaces/sinergym/.coveragerc

WORKDIR /workspaces/sinergym

CMD ["python", "scripts/try_env.py"]