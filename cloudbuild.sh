#!/bin/bash

# Default values
PROJECT_ID="sinergym"
REGION="europe-west1"
REPOSITORY="sinergym"
IMAGE_NAME="sinergym-image"
TAG="latest"
EXTRAS="drl gcloud"

# Override default values with provided arguments
while getopts p:r:R:i:t:e: flag
do
    case "${flag}" in
        p) PROJECT_ID=${OPTARG};;
        r) REGION=${OPTARG};;
        R) REPOSITORY=${OPTARG};;
        i) IMAGE_NAME=${OPTARG};;
        t) TAG=${OPTARG};;
        e) EXTRAS=${OPTARG};;
    esac
done

# Authenticate with Google Cloud
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build the container image
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG} --build-arg "SINERGYM_EXTRAS=${EXTRAS}" .

# Push the image to Artifact Registry
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}