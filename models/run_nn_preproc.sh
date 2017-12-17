#!/usr/bin/env bash
action=$1
dataset=$2
    if [[ $action == '' ]]; then
    echo "Must specify action: w2e_map/int_map"
    exit 1
fi
if [[ $dataset == '' ]]; then
    echo "Must specify dataset: grec/ms27k"
    exit 1
fi

# $CUR_PROJ_ROOT is a environment variable; manually set outside of the script.
log_path="$CUR_PROJ_ROOT/logs/models"
mkdir -p $log_path
if [[ $dataset == 'grec' ]]; then
    embeddings_path="$CUR_PROJ_ROOT/datasets/embeddings/glove"
elif [[ $dataset == 'ms27k' ]]; then
    embeddings_path="$CUR_PROJ_ROOT/datasets/embeddings/msew2v"
else
    echo "Unknown argument: $dataset"
    exit 1
fi
splits_path="$CUR_PROJ_ROOT/processed/${dataset}-naryus"

script_name="nn_preproc"
source_path="$CUR_PROJ_ROOT/source/"

if [[ $action == 'w2e_map' ]]; then
    cmd="python2 -u $source_path/models/$script_name.py $action --embeddings_path $embeddings_path --dataset $dataset"
    echo $cmd | tee "$log_path/${script_name}_${action}_${dataset}_logs.txt"
    eval $cmd | tee -a "$log_path/${script_name}_${action}_${dataset}_logs.txt"
elif [[ $action == 'int_map' ]]; then
    sizes=("small" "full")
    for size in ${sizes[@]}; do
        cmd="python2 -u $source_path/models/$script_name.py $action --in_path $splits_path --size $size"
        echo $cmd | tee "$log_path/${script_name}_${action}_${dataset}_logs.txt"
        eval $cmd | tee -a "$log_path/${script_name}_${action}_${dataset}_logs.txt"
        echo '' | tee -a "$log_path/${script_name}_${action}_${dataset}_logs.txt"
    done
fi
