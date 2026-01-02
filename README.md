# Uni-NaVid 

**A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks.** This project contains the finetuning and evaluation code of our RSS 2025 paper.


Contributors: [Jiazhao Zhang](https://jzhzhang.github.io/), Kunyu Wang, [Shaoan Wang](https://wsakobe.github.io/), Minghan Li, [Haoran Liu](https://yiconghong.me/), [Songlin Wei](https://songlin.github.io/), [Zhongyuan Wang](https://www.wangzhongyuan.com/), [Zhizheng Zhang](https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en), [He Wang](https://hughw19.github.io/)<br>

[[Paper & Appendices](https://arxiv.org/pdf/2412.06224)] [[Projece Page](https://pku-epic.github.io/Uni-NaVid/)]



<!-- https://github.com/user-attachments/assets/4ee1f806-03bb-4fcb-828e-2a7d9c6620c9



https://github.com/user-attachments/assets/304a512f-bfac-46e2-b293-f2e1e8b04f63 -->

![pipeline](./assets/uninavid.png)

## Execution
```
docker run -it \
    --name merged_navid_container \
    --privileged \
    --net=host \
    -v /dev:/dev \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/aprl/Desktop/when2reason:/when2reason \
    -w /when2reason \
    uni-navid:merged
```
- reconnect the container
```
# 1. 컨테이너 시작
docker start merged_navid_container

# 2. 실행 중인 컨테이너에 접속
docker exec -it merged_navid_container bash
```

## Reproduce result (vln/test_case)
```
Total 36 images
step 1 inference time 1.739469051361084
step 2 inference time 1.7401456832885742
step 3 inference time 1.5051002502441406
step 4 inference time 1.9687855243682861
step 5 inference time 1.726628065109253
step 6 inference time 0.947462797164917
step 7 inference time 1.0341215133666992
step 8 inference time 1.5056328773498535
step 9 inference time 1.0341827869415283
step 10 inference time 1.7389788627624512
step 11 inference time 1.3480875492095947
step 12 inference time 2.522409200668335
step 13 inference time 84.17446517944336
step 14 inference time 2.2372026443481445
step 15 inference time 1.1967601776123047
step 16 inference time 1.2066388130187988
step 17 inference time 0.4810328483581543
step 18 inference time 1.7357473373413086
step 19 inference time 17.977401733398438
step 20 inference time 1.2808616161346436
step 21 inference time 85.0175130367279
step 22 inference time 4.474096298217773
step 23 inference time 0.8890559673309326
step 24 inference time 1.3718831539154053
step 25 inference time 3.900692939758301
step 26 inference time 84.92400217056274
step 27 inference time 0.4976844787597656
step 28 inference time 1.2888250350952148
step 29 inference time 1.6021029949188232
step 30 inference time 0.4815680980682373
step 31 inference time 85.26530909538269
step 32 inference time 2.0947341918945312
step 33 inference time 85.42537641525269
step 34 inference time 2.8887200355529785
step 35 inference time 3.2006418704986572
step 36 inference time 3.664170026779175

```
## Cotents 

- [Install](#Install)
- [Preparation](#Preparation)
    - Model Preparation
    - Data Preparation
- [Train](#Train)
- [Evaluation](#Evaluation)
    - Offline Evaluation
    - Benchmark Evaluation
- [Citation](#Citation)
- [Acknowledgments](#Acknowledgments)


## Install

First, clone this repo:
```
git@github.com:jzhzhang/Uni-NaVid.git
```
Then install the Package and dependences:
```
conda create -n uninavid python=3.10 -y
conda activate uninavid
cd Uni-NaVid
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
Finall, install the flash-attn package:
```
pip install flash-attn==2.5.9.post1
```

## Preparation

### Model

To train our model, you need to download the vision encoder and the language model. Below are the links to download the models in our paper:

| Model type | Model name | Download | 
|------|------|------|
| Encoder | EVA-CLIP | [ckpt](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)|
| Pretrain model | Vicuna-7B | [ckpt](https://huggingface.co/lmsys/vicuna-7b-v1.5)|
| Finetuned model | Uni-NaVid (7B) | [ckpt](https://huggingface.co/Jzzhang/Uni-NaVid/tree/main/uninavid-7b-full-224-video-fps-1-grid-2)|

### Data

We provide a small subset of the data used in our paper to facilitate quick reproduction and customization with your own data. The data can be downloaded from [here](https://huggingface.co/Jzzhang/Uni-NaVid/tree/main/Nav-Finetune). The data is collcted from navigation tasks including the training splits of [VLN-CE](https://github.com/jacobkrantz/VLN-CE) R2R and RxR, [EVT-Bench](https://github.com/wsakobe/TrackVLA), [ObjectNav](https://arxiv.org/abs/2006.13171), [EQA](https://embodiedqa.org/). **Note that due to licensing restrictions, we did not use the [L3MVN](https://arxiv.org/pdf/2304.05501) method for ObjectNav limiation learning, which may result in a slight performance drop in ObjectNav evaluation.**

We recommend organizing your project directory as follows
```
Uni-NaVid
├── data
    ├── Nav-Finetune
        ├── nav_videos
        ├── open_uninavid_sampled_500.json
├── model_zoo
    ├── eva_vit_g.pth
    ├── <vicuna_weights> # optinoal, if you want to finetune from vicuna
    ├── <uninavid_weights> 
├── scripts
├── uninavid
├── test_cases # optinoal, if you want to offline evaluate uni-navid
```

## Train

Please set the `DATA_PATH` and `MODEL_PATH` in the `uninavid_stage_1.sh` and `uninavid_stage_2.sh` scripts to your data and model paths.

If you want to finetune from Vicuna-7B (make sure you collect sufficient data):
```
bash uninavid_stage_1.sh
```

If you want to  finetune based on Uni-NaVid:
```
bash uninavid_stage_2.sh
```


## Evaluation
During evaluation, the model leverages online token merging (`run_type=eval`), achieving an inference speed of approximately 5 Hz on a single A100 GPU. By employing more advanced techniques, such as quantization, the speed can be further enhanced.


### Offline Evaluation
We provide the offline evaluation code of Uni-NaVid on real-world videos, including a VLN sample `vln_1` and a tracking sample `tracking_1`. You can download the sample videos from [here](https://huggingface.co/Jzzhang/Uni-NaVid/tree/main/test_cases).

```
python offline_eval_uninavid.py test_cases/vln_1 Ourpur_dir # or test_cases/tracking_1
```
https://github.com/user-attachments/assets/31592c56-8369-4389-994f-f64b151ebb59

(move to the chair, then turn left and move forward to the humanoid robot and stop.)

https://github.com/user-attachments/assets/5ae851e0-d7fd-4b29-8501-05715febfc47

(follow the man with black top and brown pants.)



### Benchmark Evaluation 
We provide the evaluation code of Uni-NaVid on VLN-CE R2R/RxR and EVT Bench. 

Find the **VLN-CE benchmark** evaluation code [here](https://github.com/jzhzhang/NaVid-VLN-CE).

| Evaliation Benchmark |  TL  |  NE  |  OS  |  SR  |  SPL |
|----------------------|:----:|:----:|:----:|:----:|:----:|
| Uni-NaVid VLN-CE R2R Val.      | 9.22 | 4.96 | 57.4 | 51.8 | 47.7 |
| Uni-NaVid VLN-CE RxR Val.      | 18.4 | 5.67 | 66.4 | 56.1 | 44.5 |

Find the **EVT-bench** evaluation code [here](https://github.com/wsakobe/TrackVLA).

| Evaliation Benchmark |  SR  |  TR  |  CR  | 
|----------------------|:----:|:----:|:----:|
| Uni-NaVid EVT-Bench STT  | 53.3 | 67.2 | 12.6 | 
| Uni-NaVid EVT-Bench DT  | 31.9 | 50.1 | 21.3 | 
| Uni-NaVid EVT-Bench AT   | 15.8 | 41.5 | 26.5 | 


## Citation
If you find this work useful for your research, please consider citing:
```
@article{zhang2024uni,
    title={Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks},
    author={Zhang, Jiazhao and Wang, Kunyu and Wang, Shaoan and Li, Minghan and Liu, Haoran and Wei, Songlin and Wang, Zhongyuan and Zhang, Zhizheng and Wang, He},
    journal={Robotics: Science and Systems},
    year={2025}
}
```



## Acknowledgments
Our code is based on [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID) and [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE). 

This is an open-source version of Uni-NaVid, some functions have been rewritten to avoid certain license. 

If you have any questions, feel free to email Jiazhao Zhang at zhngjizh@gmail.com.
