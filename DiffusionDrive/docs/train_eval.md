# DiffusionDrive Training and Evaluation

## 1. Download the stage-1 ckpt from SparseDrive
```bash
wget https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1.pth
```


## 2. Training


```bash
export WORK_DIR="/job_data/work_dirs/"
export GPUS=8
export CONFIG="./projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py"
python -m torch.distributed.run \
    --nproc_per_node=${GPUS} \
    --master_port=2333 \
    tools/train.py ${CONFIG} \
    --launcher pytorch \
    --deterministic \
    --work-dir ${WORK_DIR}

```

## 3. Evaluation

```bash
bash ./tools/dist_test.sh \
    projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
    ckpt/diffusiondrive_stage2.pth \
    8 \
    --deterministic \
    --eval bbox
```

