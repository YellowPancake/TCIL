#!/usr/bin/env bash
name='cifar_b50_2s'
debug='1'
comments='None'
expid='cifar_b50_2s'


if [ ${debug} -eq '0' ]; then
    python -m main train with "./configs/${expid}.yaml" \
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
    python -m main train with "./configs/${expid}.yaml" \
        exp.name="${name}" \
        exp.savedir="./logs/" \
        exp.saveckpt="./ckpts_${expid}/" \
        exp.ckptdir="./logs/" \
        exp.tensorboard_dir="./tensorboard/" \
        exp.debug=True \
        --name="${name}" \
        -D \
        -p \
        --force \
        #--mongo_db=10.10.10.100:30620:debug
fi
