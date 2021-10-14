#!/bin/sh

#Create account service for storage
gcloud iam service-accounts create storage-account
# Grants permissions for the count
gcloud projects add-iam-policy-binding sinergym --member="serviceAccount:storage-account@sinergym.iam.gserviceaccount.com" --role="roles/owner"
# Generate key file 
gcloud iam service-accounts keys create google-storage.json --iam-account=storage-account@sinergym.iam.gserviceaccount.com