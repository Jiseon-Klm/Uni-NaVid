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

- inference time's vary according to visualization module 
```
Total 36 images
step 1 inference time 7.636366128921509
step 2 inference time 9.774767637252808
step 3 inference time 8.606340885162354
step 4 inference time 11.943239688873291
step 5 inference time 9.865652322769165
step 6 inference time 5.315337657928467
step 7 inference time 5.859358072280884
step 8 inference time 8.585879564285278
step 9 inference time 5.81484055519104
step 10 inference time 10.184200763702393
step 11 inference time 7.460430145263672
step 12 inference time 14.389869451522827
step 13 inference time 477.1044900417328
step 14 inference time 12.95702600479126
step 15 inference time 7.424868106842041
step 16 inference time 7.13951301574707
step 17 inference time 2.7361743450164795
step 18 inference time 10.124523639678955
step 19 inference time 104.55162692070007
step 20 inference time 7.546741008758545
step 21 inference time 473.1288175582886
step 22 inference time 25.734736680984497
step 23 inference time 5.09380578994751
step 24 inference time 7.600756645202637
step 25 inference time 22.878466844558716
step 26 inference time 470.30569982528687
step 27 inference time 2.7298636436462402
step 28 inference time 7.273933172225952
step 29 inference time 9.266916513442993
step 30 inference time 2.83697772026062
step 31 inference time 478.6643319129944
step 32 inference time 12.443191289901733
step 33 inference time 480.97386264801025
step 34 inference time 17.15666174888611
step 35 inference time 18.686693906784058
step 36 inference time 21.392287015914917

```

## Evaluation
During evaluation, the model leverages online token merging (`run_type=eval`), achieving an inference speed of approximately 5 Hz on a single A100 GPU. By employing more advanced techniques, such as quantization, the speed can be further enhanced.

### Online Evaluation (1fps)
```
python online_eval_uninavid.py --instruction "your task" --save_gif
```

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

## Note
- if i don't give the navigation token, it will outputs the language token not an low-level action,
<img width="1406" height="132" alt="image" src="https://github.com/user-attachments/assets/097b9766-cac9-4a3e-991e-b0465c9b86ac" />
<img width="736" height="333" alt="image" src="https://github.com/user-attachments/assets/a04308db-82c8-455d-9657-6001295a622e" />

If you have any questions, feel free to email Jiazhao Zhang at zhngjizh@gmail.com.

```
Initialization Complete
Total 12 images
step 1 inference time 1.7787134647369385
['nobody.', "I'm", 'just', 'saying', 'that', "it's", 'a', 'possibility.']
step 2 inference time 1.756117343902588
['nobody', 'is', 'going', 'to', 'stop', 'us.', "We'll", 'just', 'keep', 'going', 'until', 'we', 'get', 'to', 'the', 'other', 'side.']
step 3 inference time 1.508873701095581
['nobody', 'is', 'going', 'to', 'do', 'anything', 'about', 'it.', "I's", 'just', 'a', 'waste', 'of', 'time.']
step 4 inference time 1.9979968070983887
['nobody.', "I'm", 'just', 'saying', 'that', "I'm", 'not', 'sure', 'if', "it's", 'a', 'good', 'idea', 'to', 'do', 'that.']
step 5 inference time 1.7468583583831787
['nobody', 'can', 'stop', 'the', 'flow', 'of', 'information', 'in', 'the', 'digital', 'age.', '(the', 'internet,', 'social', 'media,', 'etc.).']
step 6 inference time 0.9617347717285156
['nobody.', "I'm", 'just', 'trying', 'to', 'help', 'you.']
step 7 inference time 1.040766716003418
['nobody', 'is', 'going', 'to', 'do', 'that.', '(laughs)']
step 8 inference time 1.513587236404419
['nobody', 'is', 'going', 'to', 'do', 'anything', 'about', 'it.', "I's", 'just', 'a', 'waste', 'of', 'time.']
step 9 inference time 1.0383918285369873
['nobody', 'can', 'stop', 'us', 'now.', 'we', 'are', 'on', 'a', 'roll.']
step 10 inference time 1.7520105838775635
['nobody', 'is', 'going', 'to', 'stop', 'us.', "We's", 'going', 'to', 'keep', 'going', 'until', 'we', 'get', 'what', 'we', 'want.']
step 11 inference time 1.3607921600341797
['nobody', 'is', 'going', 'to', 'be', 'ables', 'to', 'do', 'that.', '(laughter)']
step 12 inference time 2.5440990924835205
['nobody', 'is', 'going', 'to', 'pay', 'for', 'that.', '(the', 'cost', 'of', 'the', 'product)', 'and', 'the', 'only', 'way', 'to', 'make', 'money', 'is', 'to', 'sell', 'it', 'at', 'a', 'higher', 'price.']


```