#!/usr/bin/env bash
action=$1
which_processed=$2
if [[ $action == '' ]]; then
    echo "Must specify action: split/naryus/naryus_ent_add"
    exit 1
fi
if [[ $which_processed == '' ]] && [[ $action == 'split' ]]; then
    echo "Must specify which processed data: grec-readable"
    exit 1
fi
# $CUR_PROJ_ROOT is a environment variable; manually set outside of the script.
log_path="$CUR_PROJ_ROOT/logs/pre_process"
mkdir -p $log_path

script_name="pre_proc_grec"
source_path="$CUR_PROJ_ROOT/source/"

out_path="$CUR_PROJ_ROOT/processed/grec-${action}"
mkdir -p $out_path
if [[ $action == 'split' ]]; then
    in_path="$CUR_PROJ_ROOT/processed/$which_processed"
    cmd="python2 -u $source_path/pre_process/$script_name.py $action \
    --in_path $in_path \
    --out_path $out_path"
elif [[ $action == 'naryus' ]]; then
    in_path="$CUR_PROJ_ROOT/processed/grec-split"
    cmd="python2 -u $source_path/pre_process/$script_name.py $action \
    --in_path $in_path \
    --out_path $out_path"
elif [[ $action == 'naryus_ent_add' ]]; then
    in_path="$CUR_PROJ_ROOT/processed/grec-naryus"
    cmd="python2 -u $source_path/pre_process/$script_name.py $action \
    --in_path $in_path"
fi

echo $cmd | tee "$log_path/${script_name}_${action}_logs.txt"
eval $cmd | tee -a "$log_path/${script_name}_${action}_logs.txt"