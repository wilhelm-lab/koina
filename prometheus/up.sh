docker run -d \
	-v $(realpath prometheus/prometheus.yml):/etc/prometheus/prometheus.yml \
    --name prometheus \
    -p 9090:9090 \
    --rm \
    prom/prometheus --config.file=/etc/prometheus/prometheus.yml
