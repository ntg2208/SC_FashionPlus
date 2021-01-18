#!/bin/sh
# -------------- PREPARE --------------------
# Make batch of input dataset
echo '***************************************** Preprocess *****************************************'
ROOT_DIR='/home/giang.nguyen/FashionStyle/FashionPlus'
LABEL_DIR=${ROOT_DIR}'/datasets/labels/' #  directory with segmentation labels
IMG_DIR=${ROOT_DIR}'/datasets/images/' # directiroy with RGB images
PICKLE_FILE=${ROOT_DIR}'/datasets/demo/test.p'
JSON_FILE=${ROOT_DIR}'/datasets/demo_dict.json'
python preprocess/prepare_input_data.py \
		   --img_dir ${IMG_DIR} \
           --mask_dir ${LABEL_DIR} \
		   --output_pickle_file ${PICKLE_FILE} \
           --output_json_file ${JSON_FILE}

# -------------- ENCODE ---------------------
# -------------- Shape - VAE ----------------------
echo '***************************************** Encode Shape *****************************************'

NZ=8
OUTPUT_NC=18
MAX_MULT=8
DOWN_SAMPLE=7
BOTNK='1d'
LAMBDA_KL=0.0001
DIVIDE_K=4

CLASS='humanparsing' # dataset name
DATAROOT=${ROOT_DIR}'/datasets/demo/'
LABEL_TXT_PATH=${ROOT_DIR}'/separate_vae/datasets/humanparsing/clothing_labels.txt'
PHASE='test'
DATASET_PARAM=${ROOT_DIR}'/separate_vae/datasets/humanparsing/garment_label_part_map.json'

python separate_vae/encode_features.py \
    --phase ${PHASE} \
    --dataroot ${DATAROOT} \
    --label_dir ${LABEL_DIR} \
    --label_txt_path ${LABEL_TXT_PATH} \
    --dataset_param_file ${DATASET_PARAM} \
    --name ${CLASS} \
    --share_decoder \
    --share_encoder \
    --separate_clothing_unrelated \
    --nz ${NZ} \
    --output_nc ${OUTPUT_NC} \
    --use_dropout \
    --lambda_kl ${LAMBDA_KL}\
    --max_mult ${MAX_MULT} \
    --n_downsample_global ${DOWN_SAMPLE} \
    --bottleneck ${BOTNK} \
    --resize_or_crop pad_and_resize \
    --loadSize 256 \
    --batchSize 1 \
    --divide_by_K ${DIVIDE_K}


# -------------- Texture - pix2pix - generation --------------------
echo '***************************************** Encode Textures *****************************************'

NETG='local' # local, global
MODELNAME='pix2pixHD'

COLOR_MODE='Lab'
FEAT_NUM=8
IMG_DIR=${ROOT_DIR}'/datasets/images/'
NUM_LABEL=18
PRETRAIN=${ROOT_DIR}'/checkpoints/humanparsing'

python generation/encode_clothing_features.py \
  --dataroot ${DATAROOT} \
  --phase ${PHASE} \
  --name demo \
  --model ${MODELNAME} \
  --feat_num ${FEAT_NUM} \
  --label_feat \
  --load_pretrain ${PRETRAIN} \
  --label_dir ${LABEL_DIR} \
  --img_dir ${IMG_DIR} \
  --resize_or_crop pad_and_resize \
  --loadSize 256 \
  --label_nc ${NUM_LABEL} \
  --color_mode ${COLOR_MODE}

# -------------- EDIT --------------------
echo '***************************************** Edit *****************************************'

NET_ARCH='mlp' # linear, mlp
MODEL='pix2pixHD'
LAMBDA_KL=0.0001 # hyperparameter for VAE 0.0001
DIVIDE_K=4 # hyperparameter for VAE 4
TEXTURE_FEAT_NUM=8

# Editing module options
UPDATE_FNAME='998.jpg' # filename to update 18.jpg
UPDATE_TYPE='texture_only' # specify whether to edit shape_only, texture_only, or **shape_and_texture**
SWAPPED_PARTID=1 # swapped_partID specifies which part to update; for class='humanparsing', partID mapping is: 0 top, 1 skirt, 2 pants, 3 dress 0
MAXITER=10 # editing module stops at maxtier iterations 10
UPDATE_STEP_SZ=0.25 # editing module takes step size at each update iteration 0.25
ITERATIVE_SAVE='False' # iterative_save is True when we generate edited resutls from each iteration

DFEAT=64
PARAM_M=3
PARAM_K=256
CLF_EPOCH=120
DATASET_DIR=${ROOT_DIR}'/datasets/'
SAVE_DIR='results/demo/'
TEXTURE_PATH=${ROOT_DIR}'/generation/results/Lab/demo/test_features.p'
TEXTURE_GEN_PATH=${ROOT_DIR}'/checkpoints/'
SHAPE_PATH=${ROOT_DIR}'/separate_vae/results/Lab/demo/test_shape_codes.p'
SHAPE_GEN_PATH=${ROOT_DIR}'/checkpoints/'
CLASSIFIER_PATH=${ROOT_DIR}'/checkpoints/m'${PARAM_M}'k'${PARAM_K}'/'

############### UPDATE AND GENERATE  ###############
python classification/data_dict/shape_and_feature/update_demo.py \
        --update_fname ${UPDATE_FNAME} \
        --update_type ${UPDATE_TYPE} \
        --max_iter_hr ${MAXITER} \
        --swapped_partID ${SWAPPED_PARTID} \
        --lr ${UPDATE_STEP_SZ} \
        --min_thresholdloss 0.00009 \
        --model_type ${MODEL} \
        --texture_feat_num ${TEXTURE_FEAT_NUM} \
        --texture_feat_file ${TEXTURE_PATH} \
        --shape_feat_file ${SHAPE_PATH} \
        --dataset_dir ${DATASET_DIR} \
        --param_m ${PARAM_M} \
        --param_k ${PARAM_K} \
        --load_pretrain_clf ${CLASSIFIER_PATH} \
        --load_pretrain_texture_gen ${TEXTURE_GEN_PATH} \
        --load_pretrain_shape_gen ${SHAPE_GEN_PATH} \
        --network_arch ${NET_ARCH} \
        --in_dim ${DFEAT} \
        --clf_epoch ${CLF_EPOCH} \
        --lambda_smooth 0 \
        --display_freq 1 \
        --classname ${CLASS} \
        --color_mode ${COLOR_MODE} \
        --save_dir ${SAVE_DIR} \
        --generate_or_save generate