#!/usr/bin/env bash
action=$1
dataset=$2
if [[ $action == '' ]]; then
    echo "Must specify action: nearest_ents/nearest_row/nearest_col"
    exit 1
fi
if [[ $dataset == '' ]]; then
    echo "Must specify dataset: grec/ms27k"
    exit 1
fi

if [[ $dataset == 'grec' ]]; then
    int_mapped_path="$CUR_PROJ_ROOT/processed/grec-naryus"
    run_path="$CUR_PROJ_ROOT/model_runs/train_model-grec-2017_12_18-16_04_57"
elif [[ $dataset == 'ms27k' ]]; then
    int_mapped_path="$CUR_PROJ_ROOT/processed/ms27k-naryus"
    run_path="$CUR_PROJ_ROOT/model_runs/train_model-ms27k-2017_12_18-15_36_11"
else
    echo "Unknown argument: $dataset"
    exit 1
fi

log_path="$CUR_PROJ_ROOT/logs/models"
script_name="man_eval"
source_path="$CUR_PROJ_ROOT/source/"


if [[ $action == 'nearest_ents' ]] || [[ $action == 'nearest_col' ]] || [[ $action == 'nearest_row' ]]; then
    cmd="python2 -u $source_path/models/$script_name.py $action --int_mapped_path $int_mapped_path --run_path $run_path"
    echo $cmd | tee "$log_path/${script_name}_${action}_${dataset}_logs.txt"
    eval $cmd | tee -a "$log_path/${script_name}_${action}_${dataset}_logs.txt"
else
    echo "Unknown action"
    exit 1
fi