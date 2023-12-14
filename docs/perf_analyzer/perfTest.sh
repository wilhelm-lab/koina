N=1
(
 for p in $(cat models.txt); do
    ((i=i%N)); ((i++==0)) && wait
    perf_analyzer --measurement-interval 50000 -u 'serving:8501' -m $p --concurrency-range 50 -b 1000 --input-data /workspace/koina/docs/perf_analyzer/input.json --shape 'peptide_sequences:1' > log/$p.log 2>&1 &
 done
)
