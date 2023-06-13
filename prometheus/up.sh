docker run -d \
    -v /home/user02/koina/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
    --name prometheus \
    --network host \
    prom/prometheus --web.listen-address=:8501 --config.file=/etc/prometheus/prometheus.yml