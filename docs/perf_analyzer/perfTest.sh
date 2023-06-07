while read p; do
  docker-compose exec -d develop perf_analyzer -u 'serving:8501' -m $p --concurrency-range 500 -b 1000
done <models.txt
