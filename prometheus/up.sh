docker run -d \
    -v $(realpath prometheus/prometheus.yml):/etc/prometheus/prometheus.yml \
    -v /var/log/prometheus/data:/prometheus/data \
    --name prometheus \
    -p 9090:9090 \
    prom/prometheus --config.file=/etc/prometheus/prometheus.yml
