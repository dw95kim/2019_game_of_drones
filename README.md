# 2019_game_of_drones

## Env Spec
```
Ubuntu 16.04
Nvidia driver version : 410.15
```

## Requirement
```
1. python >= 3.6

2. airsimneurips >= 1.2.0

3. tensorflow - gpu >= 1.14.0

4. cuda >= 10.0

5. cudnn >= 7.55
```

## Installation
```
1. sh download_final_round_binaries.sh
```

## Object detection Training
```
1. generate record file (reference https://github.com/dw95kim/generate_record_file_from_raw_image)

2. object_detection/training 폴더에서 ssd_mobildnet_v1_coco.config 파일 수정
> 1. change train/test_record & labelmap file path
> 2. change checkpoint path
> 3. change the number of classes
> 4. change the number of eval data
(reference https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#3-gather-and-label-pictures)

3. 
PIPELINE_CONFIG_PATH=/home/dwkim/Desktop/LAB/object_detection/training/ssd_mobilenet_v1_coco.config
MODEL_DIR=/home/dwkim/Desktop/LAB/object_detection/training
NUM_TRAIN_STEPS=1000000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 model_main.py \
 	--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
 	--model_dir=${MODEL_DIR} \
 	--num_train_steps=${NUM_TRAIN_STEPS} \
 	--sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
 	--alsologtostderr

```

## Export inference graph
- you also use 'tensorboard --logdir=training'
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph

```

## Excution
```
 1. ./AirSimExe.sh -windowed -opengl4
 
 2. python baseline_racer_tier2_final_round.py
 (or baseline_racer_tier3_final_round.py, baseline_racer_train_one_by_one_upgrade.py)
```
