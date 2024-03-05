#!/bin/bash
#Healthcheck serving
url="serving:8501/v2/health/ready"
interval=5
max_attempts=120
timeout=1

echo "Waiting for serving to start!"
for ((attempt=1; attempt<=$max_attempts; attempt++)); do
    echo "Attempt $attempt/$max_attempts:"
    http_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time $timeout $url)

    if [ "$http_status" -eq 200 ]; then
        echo "Success! Received 200 status code."
        break
    fi

    echo "Received $http_status status code. Retrying in $interval seconds..."
    sleep $interval
done