python train.py \
MODEL.NAME transformer \
MODEL.PRETRAIN_CHOICE imagenet \
SOLVER.BASE_LR 0.0004 \
DATASETS.SECOND_DIR /home/yzh_92/Datasets/newClassification/opendomain_main/comparision/oridata_indomain/balance_82_plant \
MODEL.PRETRAIN_PATH /home/yzh_92/model/TransReID_res/pretrain_weights/jx_vit_base_p16_224-80ecf9dd.pth \
MODEL.CATEGORY plant \
SOLVER.IMS_PER_BATCH 96 \
DATALOADER.NUM_INSTANCE 16 \
OUTPUT_DIR ./logs/plant


python train.py \
MODEL.NAME transformer \
MODEL.PRETRAIN_CHOICE imagenet \
SOLVER.BASE_LR 0.0004 \
DATASETS.SECOND_DIR /home/yzh_92/Datasets/newClassification/opendomain_main/comparision/oridata_indomain/balance82 \
MODEL.PRETRAIN_PATH /home/yzh_92/model/TransReID_res/pretrain_weights/jx_vit_base_p16_224-80ecf9dd.pth \
MODEL.CATEGORY disease \
SOLVER.IMS_PER_BATCH 96 \
DATALOADER.NUM_INSTANCE 16 \
OUTPUT_DIR ./logs/disease

python train.py \
MODEL.NAME transformer \
MODEL.PRETRAIN_CHOICE imagenet \
SOLVER.BASE_LR 0.0004 \
DATASETS.SECOND_DIR /home/yzh_92/Datasets/newClassification/opendomain_main/comparision/oridata_indomain/balance_82_severity \
MODEL.PRETRAIN_PATH /home/yzh_92/model/TransReID_res/pretrain_weights/jx_vit_base_p16_224-80ecf9dd.pth \
MODEL.CATEGORY severity \
SOLVER.IMS_PER_BATCH 72 \
DATALOADER.NUM_INSTANCE 24 \
OUTPUT_DIR ./logs/severity