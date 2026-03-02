#!/usr/bin/env bash
set -e  
set -u
set -o pipefail

# ===== Parameters Settings =====
bin_path="./build/examples/demo"                
method="PQ(4,6,16)"                    # PQ(nc,nb,ns): nc-> P1 (number of partitions), number of bits per subspace, number of subspaces
search_topn=2                           # probe partitions
nprobe=24                               # probe local lists (apply adaptive probing by default)
nlists=1024                             # P2 in the paper, number of local lists
name="siftsmall"                       # name of the dataset, used in metric files
topk=100                            # topk retrieval
refine=200       # Support up to 400 now, will extend in future
                 # number of clusters to probe, must less or equal than number of clusters
search_thread=1
trial=1                          # run query trial times, each trial run repeat times. Use faster latency in each trial, and average across all trials
repeat=1                            
metric_file="metric/${name}_${method}.csv" # will record all running metrics
metric_head="Dataset,Imbalance,NumCentroids,NumSubspaces,NumClusters,TopNClusters,NLists,NProbe,SearchThreads,UpSample,Refine,Simd,QuantBit,Throughput(Q/sec),Latency,MemMB,TrainSec,Recallk@k,k"

dataset_dir="dataset"
if [ -z "$dataset_dir" ]; then
    echo "Please Set Your Dataset Dir At (\${dataset_dir})"
    exit 1
fi

base="${dataset_dir}/${name}_base.fvecs" # database vectors path, in fvecs format
query="${dataset_dir}/${name}_query.fvecs" # query vectors path, in fvecs format
gt="${dataset_dir}/${name}_groundtruth.ivecs"  # groundtruth vectors path, in ivecs format
dim=128
base_size=10000 # number of database vectors
query_size=100 # number of query vectors
file_format_ori="fvecs"
groundtruth_format="ivecs"

# output path
answer_dir="answer" # path to output retreival vectors results
encoded_dir="encoded_dataset" # path to store the PQ index
mkdir -p "$answer_dir" "$encoded_dir" "$(dirname "$metric_file")"

if [[ ! -f "$metric_file" ]]; then
    echo "Creating $metric_file ..."
    echo "$metric_head" > "$metric_file"
else
    echo "$metric_head already exists, skip."
fi

# ===== Execution =====
${bin_path} \
    --dataset "${base}" \
    --queries "${query}" \
    --file-format-ori "${file_format_ori}" \
    --dim "${dim}" \
    --dataset-size "${base_size}" \
    --queries-size "${query_size}" \
    --result "${answer_dir}/answer_${method}_${name}.csv" \
    --groundtruth "${gt}" \
    --groundtruth-format "${groundtruth_format}" \
    --method "${method}" \
    --k "${topk}" \
    --refine "${refine}" \
    --save "${encoded_dir}/${name}_${method}_${nlists}.bin" \
    --search-topn "${search_topn}" \
    --config-id "${name}" \
    --trial "${trial}" \
    --metric-log "${metric_file}" \
    --repeat "${repeat}" \
    --r-at-r "1" \
    --centroid_distribution "0" \
    --alpha "1.0" \
    --nprobe "${nprobe}" \
    --nlist "${nlists}" \
    --search_thread "${search_thread}" \
    --up-sample "1"





echo "Done: ${name} ${method}"
