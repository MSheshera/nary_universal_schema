#!/usr/bin/env bash

action=$1
dataset=$2
run_path=$3
if [[ $action == '' ]]; then
    echo "Must specify action: run_saved/train_model"
    exit 1
fi
if [[ $dataset == '' ]]; then
    echo "Must specify dataset: grec/ms27k"
    exit 1
fi
if [[ $run_path == '' ]] && [[ $action == 'run_saved' ]] ; then
    echo "Must specify full path to trained model."
    exit 1
fi

edim=100
hdim=50
dropp=0.3
lr=0.01
run_time=`date '+%Y_%m_%d-%H_%M_%S'`
if [[ $dataset == 'grec' ]]; then
    embedding_path="$CUR_PROJ_ROOT/datasets/embeddings/glove"
    int_mapped_path="$CUR_PROJ_ROOT/processed/grec-naryus"
elif [[ $dataset == 'ms27k' ]]; then
    embedding_path="$CUR_PROJ_ROOT/datasets/embeddings/msew2v"
    int_mapped_path="$CUR_PROJ_ROOT/processed/ms27k-naryus"
else
    echo "Unknown argument: $dataset"
    exit 1
fi

if [[ $action == 'train_model' ]]; then
    run_name="${action}-${dataset}-${run_time}"
    run_path="$CUR_PROJ_ROOT/model_runs/${run_name}"
    mkdir -p $run_path
    cmd="python2 -u main.py  train_model --int_mapped_path $int_mapped_path \
         --embedding_path $embedding_path \
         --run_path $run_path \
         --edim $edim --hdim $hdim --dropp $dropp \
         --bsize 64 --epochs 3 --lr $lr"
    echo $cmd | tee "$run_path/train_run_log.txt"
    eval $cmd | tee -a "$run_path/train_run_log.txt"
    # Move the generated plots to a result dir for easy rsync to local.
    mkdir -p "$CUR_PROJ_ROOT/rsync_results/${run_name}"
    cp -l "$run_path/"*.png "$CUR_PROJ_ROOT/rsync_results/${run_name}"
elif [[ $action == 'run_saved' ]]; then
    cmd="python2 -u main.py  run_saved_model --int_mapped_path $int_mapped_path \
         --embedding_path $embedding_path \
         --run_path $run_path"
    echo $cmd | tee "$run_path/saved_run_log.txt"
    eval $cmd | tee -a "$run_path/saved_run_log.txt"
fi