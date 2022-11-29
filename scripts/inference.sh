#!/usr/bin/env bash
name='Inference_cifar_b0_10s'
debug='1'
comments='None'
expid='cifar_b0_10s'


if [ ${debug} -eq '0' ]; then
    python -m main test with "./configs/${expid}.yaml" \
        exp.name="${name}" \
        exp.savedir="./logs/" \
        exp.ckptdir="./logs/" \
        exp.saveckpt="./ckpts_${expid}/" \
        exp.tensorboard_dir="./tensorboard/" \
        trial=0 \
        --name="${name}" \
        -D \
        -p \
        -c "${comments}" \
        --force \
        --mongo_db=10.10.10.100:30620:classil
        # --mongo_db=10.10.10.100:30620:classil
else
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
        #--mongo_db=10.10.10.100:30620:debug
fi
