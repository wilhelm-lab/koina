N=13
(
 for p in $(cat models.txt); do
    ((i=i%N)); ((i++==0)) && wait
    perf_analyzer -u 'serving:8501' -m $p --concurrency-range 1 -b 10 --input-data /workspace/koina/docs/perf_analyzer/input.json --shape 'peptide_sequences:1' > log/$p.log 2>&1 &
 done 
)