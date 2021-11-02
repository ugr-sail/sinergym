#!/bin/bash

# This scrip is used to build a mlflow server in Google Cloud, it is important
# to set up account previously.

# Please, visit our documentation here --> https://jajimer.github.io/sinergym/build/html/index.html

# Step 0 - Store all parameters

PROJECT_ID=$1
BUCKET_NAME=$2
ZONE=$3
REGION=${ZONE::-2}
DB_ROOT_PASSWORD=$4
MACHINE_TYPE=e2-medium
MLFLOW_IMAGE=kaysush/mlflow:1.14.1
CLOUD_SQL_PROXY_IMAGE=gcr.io/cloudsql-docker/gce-proxy:1.19.1
MYSQL_INSTANCE=${PROJECT_ID}:${REGION}:mlflow-backend

# Step 1 - Service account for mlflow service
echo "Creating Service account for mlflow service [mlflow-tracking-sa]..."
gcloud iam service-accounts create mlflow-tracking-sa --description="Service Account to run the MLFLow tracking server" --display-name="MLFlow tracking SA"

# Step 2 - Artifact used by mlflow to store all runs information
echo "Creating Back-end artifact bucket [$BUCKET_NAME]..."
gsutil mb gs://$BUCKET_NAME

# Step 3 - CLoud SQL, instance with SQL and "mlflow" database inner
echo "Creating sql instance with mlflow database [mlflow-backend]..."
gcloud sql instances create mlflow-backend --tier=db-f1-micro --region=${REGION} --root-password=${DB_ROOT_PASSWORD} --storage-type=SSD
gcloud sql databases create mlflow --instance=mlflow-backend

# Step 4 - IAM: Provisioning service account privileges in order to manipulate bucket and back-end
echo "Creating service account privileges to use Back-end [roles/cloudsql.editor]..."
gsutil iam ch "serviceAccount:mlflow-tracking-sa@${PROJECT_ID}.iam.gserviceaccount.com:roles/storage.admin" gs://${BUCKET_NAME}
gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="serviceAccount:mlflow-tracking-sa@${PROJECT_ID}.iam.gserviceaccount.com" --role=roles/cloudsql.editor

# Step 5 - Creating start_mlflow_tracking.sh to initialize instance
echo "Creating start_mlflow_tracking.sh to initialize instance..."
cat <<EOF >./start_mlflow_tracking.sh
echo "Starting Cloud SQL Proxy'"
docker run -d --name mysql  --net host -p 3306:3306 $CLOUD_SQL_PROXY_IMAGE /cloud_sql_proxy -instances $MYSQL_INSTANCE=tcp:0.0.0.0:3306

echo "Starting mlflow-tracking server"
docker run -d --name mlflow-tracking --net host -p 5000:5000 $MLFLOW_IMAGE mlflow server --backend-store-uri mysql+pymysql://root:${DB_ROOT_PASSWORD}@localhost/mlflow --default-artifact-root gs://${BUCKET_NAME}/mlflow_artifacts/ --host 0.0.0.0

echo "Altering IPTables"
iptables -A INPUT -p tcp --dport 5000 -j ACCEPT
EOF

# Step 6 - Uploading start script and deleting from local
echo "Uploading start_mlflow_tracking.sh at gs://${BUCKET_NAME}/scripts/start_mlflow_tracking.sh..."
gsutil cp ./start_mlflow_tracking.sh gs://${BUCKET_NAME}/scripts/start_mlflow_tracking.sh
echo "Deleting temporal local script [start_mlflow_tracking.sh]"
rm ./start_mlflow_tracking.sh

#Step 7 - creating static external ip for mlflow server
echo "Creating static external ip for mlflow-tracking-server [mlflow-ip]"
gcloud compute addresses create mlflow-ip \
    --region europe-west1

# Step 8 - Compute Instance
echo "Deploying remote server [mlflow-tracking-server]..."
gcloud compute --project=$PROJECT_ID instances create mlflow-tracking-server \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --subnet=default \
    --network-tier=PREMIUM \
    --metadata=startup-script-url=gs://${BUCKET_NAME}/scripts/start_mlflow_tracking.sh \
    --maintenance-policy=MIGRATE \
    --service-account=mlflow-tracking-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=mlflow-tracking-server \
    --image=cos-77-12371-1109-0 \
    --image-project=cos-cloud \
    --boot-disk-size=10GB \
    --boot-disk-type=pd-balanced \
    --boot-disk-device-name=mlflow-tracking-server \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --reservation-affinity=any \
    --address $(gcloud compute addresses describe mlflow-ip --format='get(address)')

# Step 8 - Firewall
echo "Creating firewall rules [allow-mlflow-tracking]..."
gcloud compute firewall-rules create allow-mlflow-tracking --network default --priority 1000 --direction ingress --action allow --target-tags mlflow-tracking-server --source-ranges 0.0.0.0/0 --rules tcp:5000 --enable-logging

