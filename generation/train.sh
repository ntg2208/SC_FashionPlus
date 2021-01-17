PATH_TO_SAVE_MODEL=../checkpoints/
PATH_TO_SEGMENTATION_MAPS=../datasets/humanparsing/train_label/
PATH_TO_IMG=../datasets/humanparsing/train_img
python ./train.py \
    --dataroot ../datasets/humanparsing \
    --name humanparsing \
    --label_feat \
    --checkpoints_dir ${PATH_TO_SAVE_MODEL} \
    --label_dir ${PATH_TO_SEGMENTATION_MAPS} \
    --img_dir ${PATH_TO_IMG} \
    --resize_or_crop pad_and_resize \
    --loadSize 256 \
    --fineSize 512 \
    --save_epoch_freq 100 \
    --label_nc 18 \
    --output_nc 3 \
    --color_mode Lab \
    --gpu_ids 0 \
    --batchSize 16