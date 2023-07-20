#!/bin/bash
# Path: HAT/trainmore.sh
# a shell script to run following:

python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 hat/train.py -opt ./options/train/train_HAT-L_SRx4_CCA.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 hat/train.py -opt ./options/train/train_HAT-L_SRx4_BUSI.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 hat/train.py -opt ./options/train/train_HAT-L_SRx4_BUSIBlur.yml --launcher pytorch
# a shell script to run following:
