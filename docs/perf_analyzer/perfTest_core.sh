N=13
(
 for p in $(cat models_core.txt); do
    ((i=i%N)); ((i++==0)) && wait
    perf_analyzer -u 'serving:8501' -m $p --concurrency-range 1 -b 10 > log/core/$p.log 2>&1 &
 done 
)
