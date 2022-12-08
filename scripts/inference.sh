#!/usr/bin/env bash
name='Inference_cifar_b0_10s'
expid='cifar_b0_10s'


python -m main test with "./configs/${expid}.yaml" \
    exp.name="${name}" \
    exp.savedir="./logs/" \
    exp.ckptdir="./logs/" \
    exp.saveckpt="./ckpts_${expid}/" \
    exp.tensorboard_dir="./tensorboard/" \
    exp.debug=True \
    load_mem=True \
    --name="${name}" \
    -D \
    -p \
    --force \