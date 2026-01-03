# Uni-NaVid 

**A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks.** This project contains the finetuning and evaluation code of our RSS 2025 paper.


Contributors: [Jiazhao Zhang](https://jzhzhang.github.io/), Kunyu Wang, [Shaoan Wang](https://wsakobe.github.io/), Minghan Li, [Haoran Liu](https://yiconghong.me/), [Songlin Wei](https://songlin.github.io/), [Zhongyuan Wang](https://www.wangzhongyuan.com/), [Zhizheng Zhang](https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en), [He Wang](https://hughw19.github.io/)<br>

[[Paper & Appendices](https://arxiv.org/pdf/2412.06224)] [[Projece Page](https://pku-epic.github.io/Uni-NaVid/)]



<!-- https://github.com/user-attachments/assets/4ee1f806-03bb-4fcb-828e-2a7d9c6620c9



https://github.com/user-attachments/assets/304a512f-bfac-46e2-b293-f2e1e8b04f63 -->

![pipeline](./assets/uninavid.png)

## Release
- [x] Training Code
- [x] Offline Evaluation Code
- [x] Benchmark Evalation Code
    - [x] VLN-CE
    - [x] EVT-Bench
- [x] A small split of VLN-CE RxR data


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

```
나 지금 이 논문(Uni-navid)을 구현해보고 있어. 근데 지금 내 목표는 eqa로 설정되어서 low-level action을 output하지 않고, 설명(language token)을 출력하는uni-navid의 메인 코드(offline_eval.py)를 논문의 지침에 따라 eqa가 아닌 VLN task로 인지하도록 하여 4개의 low-level action을 ouput 하도록 정확하게 input instruction을 재구성하는게 내 목표야. 지금 코드는 보다싶이 nav indicator를 아직 promt_template 에 포함하지 않았고, Your assigned task is: '{}'도 비어 있어. 그리고 문제는 논문에 나온 nav task를 구분하기위한 special indicator, nav 토큰이 코드에 구현된거랑 살짝 달라서 내가 테스트해보기위해서는 어떤 토큰을 넣어야 저자들이 의도한대로 uni-navid를 쓰는건지 헷갈려. 

헷갈리는 부분:
1. 논문에는 navigation task를 시킬거면  <NAV >토큰을 넣으라는데, 코드에는 [Navigation]이라고 되어 있어.

헷갈리는 부분을 해결해주면서 현재 eqa 모드를 논문에서 알려준대로 vln 모드로 바꾸는 법을 알려줘.



논문: 
Uni-NaVid: A Video-based Vision-Language-Action
Model for Unifying Embodied Navigation Tasks
Jiazhao Zhang1,2
Kunyu Wang3
Songlin Wei1,2
Shaoan Wang1,2
Zhongyuan Wang3
Minghan Li2
Zhizheng Zhang2,3,†
Haoran Liu1,2
He Wang1,2,3,†
1CFCS, School of Computer Science, Peking University 2Galbot 3Beijing Academy of Artificial Intelligence
https://pku-epic.github.io/Uni-NaVid
Vision-and-language Navigation
Vision-and-language Navigation
Vision-and-language Navigation
arXiv:2412.06224v2  [cs.RO]  6 Feb 2025
Training Data
Video: 
Video: 
Video: 
…
…
…
Task: Walk towards the dining 
room, wait inside the double 
doors on the left.
Task: Walk towards the dining 
room, wait inside the double 
doors on the left.
Next Actions: <Forward>...
Task: Walk towards the dining 
room, wait inside the double 
doors on the left.
Next Actions: <Forward>...
Next Actions: <Forward>...
Object Goal Navigation
Object Goal Navigation
Object Goal Navigation
Video: 
Video: 
Video: 
…
…
…
Task: Search For a chair and 
stop by it.
Task: Search For a chair and 
stop by it.
Task: Search For a chair and 
stop by it.
Next Actions: <Left>...
Next Actions: <Left>...
Next Actions: <Left>...
Embodied Question Answering
Embodied Question Answering
Embodied Question Answering
Video: 
Video: 
Video: 
…
…
…
Task: What is the room of the 
bed located in?
Task: What is the room of the 
bed located in?
Task: What is the room of the 
bed located in?
Next Actions: <Right>...
Next Actions: <Right>...
Next Actions: <Right>...
Human Following
Human Following
Human Following
Video: 
Video: 
Video: 
…
…
…
Task: Track the person
Task: Track the person
Task: Follow the person.
Next Actions: <Stop>...
Next Actions: <Stop>...
Next Actions: <Stop>...
3.6
Million
Language Instruction
Egocentric Video
Zero-shot deployment
“Move to the man on the right side. Then follow that man until you see a sofa. Turn right and search for 
a TV, stop by the TV. Finally, what is the color of the sofa?”
Uni-NaVid: Beige
Move forward 5 steps and turn left 90 
degrees. Then move to the glass door and 
turn left. And walk along the hallway, 
turn right when you near the plant. Then 
move forward and stop in front of the 
double door.
Fig. 1: Uni-NaVid learns general navigation skills across four embodied navigation tasks using 3.6 million navigation samples.
Embodied question answering
Uni-NaVid only takes online RGB video frames and language instructions as input and output actions, achieving general
Human Following
navigation ability in a real-world deployment.
Object Goal Navigation
Abstract—Embodied Navigation is a fundamental capability for
“Forward 5 steps ... move to the glass door
and turn left. And walk ... turn right when
near the plant ... stop by the double door.”
Actions 
intelligent robots, requiring robots to follow human commands
<Forward>/<Right>/<Left>/<Stop>
and move autonomously within physical environments. Despite
1.3M samples
100 scenes
“Search for a chair with 
a doll.”
“Move forward 5 steps and turn left 90 
degrees. Then move to the glass door and 
turn left. And walk along the hallway, 
turn right when you near the plant. Then 
move forward and stop in front of the 
double door.”
within a unified framework by using only ego-centric RGB
video as inputs. Additionally, real-world experiments confirm
the model’s effectiveness and efficiency, shedding light on its
significant advancements, most existing navigation approaches are
tailored to specific navigation tasks, such as instruction following,
Search for a chair with a doll 
on it.
Move out of the door and turn left, 
what is the color of the sofa?
searching objects, answering questions, tracking people, and
strong generalizability.
Search for a chair with a doll.
Follow the man with black 
jacket.
more. However, the increasing demands on advanced embodied
navigation pose the challenge of designing a practical navigation
agent that can incorporate multiple navigation tasks naturally
and benefits from the synergy between these tasks. To this end, we
present Uni-NaVid, a video-based vision-language-action (VLA)
model to unify different paradigms of navigation tasks and
improve navigation performance by encouraging the synergy
among different navigation sub-tasks. This VLA model can directly
take natural language instructions and RGB video streams as
inputs and output low-level robotic actions in an end-to-end
manner. To efficiently process extensive RGB video streams,
we propose an online token merge strategy that spatially and
I. INTRODUCTION
Embodied navigation [112, 85] is a critical capability for
intelligent robots and has drawn significant attention in the
robotics community. For successful embodied navigation,
robots must be able to move autonomously within physical
environments based on human instructions. However, nav
igation tasks vary significantly, and most existing studies
are designed for specific tasks, e.g., vision-and-language
navigation [44, 46], object goal navigation [12], embodied
question answering [21, 90], and following [113, 36, 68].
Consequently, most current approaches are developed to address
temporally consolidates similar visual information which improves
the inference speed to 5 Hz. For training Uni-NaVid, we collect
3.6 million navigation data samples across different navigation
tasks. Extensive experiments on diverse navigation benchmarks
demonstrate that Uni-NaVid achieves state-of-the-art performance
Author (e-mail: zhngjizh@gmail.com). † Corresponding authors (e-mail:
only one type of navigation task, often relying on specialized
modules and task-specific datasets. This narrow scope limits
their applicability to multi-purpose navigation applications and
prevents these methods from leveraging potential synergies
across diverse navigation tasks.
Developing a versatile navigation model presents significant
challenges, as it requires the unification of navigation task
zhangzz@galbot.com, hewang@pku.edu.cn).
modeling and the integration of heterogeneous data for joint
use. Initial efforts adopt imitation learning (IL) [85, 93, 66]
or reinforcement learning (RL) [106, 97] to learn general
navigation skills in simulation environments or limited diverse
real-world environments. However, due to the limited rendering
quality and diversity of simulators, these approaches often
encounter the “sim-to-real” gap and suffer from poor gener
alization across diverse navigation tasks [27, 5, 40]. Recent
studies [120, 114, 61, 60, 78] have attempted to achieve a
higher degree of unification using pre-trained large language
models (LLMs). However, due to the low frequency of LLM
inference, they simplify the problem to some extent by adopting
discretized modeling approaches. They rely on pre-defined
graphs for decision-making learning, which sacrifices output
f
lexibility and introduces additional challenges for real-world
deployment.
In this work, we propose Uni-NaVid, a video-based Vision
Language-Action (VLA) model for unifying diverse commonly
demanded navigation tasks (Tab. I). Uni-NaVid takes egocen
tric RGB video streams and natural language instructions as
inputs, and directly generates low-level actions for navigation
in continuous environments. To achieve multi-task navigation
while supporting efficient navigation, Uni-NaVid extend video
based VLM [51] by incoprating two key components: (1)
an efficient VLA architecture based on an online token
merge mechanism, which enables efficient processing of
online-captured video streams for LLM inference; and (2)
an extensive collection of 3.6M samples across four widely
studied navigation tasks. We provide a detailed elaboration
below:
During navigation, the agent is required to process a
substantial volume of online captured frames, which results
in memory overload and computational latency, particularly in
LLM-based approaches [111, 61]. To this end, we propose an
online token merging mechanism to compress near historical
frames with a relatively low ratio while compressing far
historical frames with a relatively high ratio. This merging
mechanism operates in an on-the-fly manner, maximizing
the reuse of previous navigation history. In this way, Uni
NaVid learn compact representations that maintain not only
f
ine-grained spatial information but also structured temporal
information, thus speeding up the model inference by reducing
the token number. Besides, Uni-NaVid adopts a foresight
prediction to generate actions for a future horizon at once
instead of step-by-step. This enables Uni-NaVid to achieve
5Hz inference, facilitating the deployment of a non-blocking
navigation robot powered by a VLA model in real-world
environments (Please refer to the supplementary video).
We aim to build Uni-NaVid as a versatile multi-task navi
gation agent, incorporating four widely demanded navigation
tasks: vision-and-language navigation, object-goal navigation,
embodied question answering, and human following. These
tasks are distinct from each other, with varying task settings
and objectives. Specifically, for the human-following task, we
construct a new language-guided human-following benchmark
Action
Methods
Embodied Navigation Tasks
D.E. C.E. VLN [44] ObjNav [76] EQA [90] Follow [68]
VLMaps [34]
✓
NaviLLM [114] ✓
InstructNav [61] ✓
Poliformer [106]
Uni-NaVid
✓
✓
✓
✓
✓
✓
✓
✓
✓
✓
✓
✓
✓
✓
✓
TABLE I: Task and setting comparison. Uni-NaVid is de
veloped to address four embodied navigation tasks, generating
action outputs in continuous environments. C.E.: Continuous
Environment; D.E.: Discrete Environment.
navigation samples based on diverse navigation tasks with
different simulation environments. Additionally, inspired by the
success of manipulation VLAs [9], we further integrate 2.3M
real-world internet data samples for Video Question Answering
(VQA) [7, 51] and video captioning [19] as auxiliary tasks. This
integration aims to enhance scene understanding and promote
sim-to-real generalization.
We conduct extensive experiments on benchmarks across
the aforementioned four navigation tasks and compared our
method with strong baselines specifically designed for each
task. Utilizing only RGB video streams and instructions as
inputs, our method demonstrates the superiority of a single
VLA model across diverse benchmarks, achieving SOTA or
SOTA-comparable performance. Furthermore, comprehensive
ablation studies validate the synergistic benefits of learning
multiple navigation tasks jointly. Finally, real-world exper
iments demonstrate that Uni-NaVid achieves non-blocking
navigation exhibiting impressive robustness in handling diverse
instructions and environments. We believe our work serves
merely as a starting point for general-purpose navigation, and
we will release the full source code to benefit the community.
II. RELATED WORKS
Multi-Task Embodied Navigation. Embodied navigation [2,
94, 112] requires agents to navigate in unseen environments
based on human instructions. There is extensive literature on
embodied navigation; here, we focus on four mainstream tasks
that involve both visual information and language instructions:
Vision-and-Language Navigation [4, 44, 46], Object Goal Nav
igation [12], Embodied Question Answering [21], and Human
Following [37, 68, 118, 119]. Early efforts [85, 93, 66, 97]
towards a generalist-embodied navigation model involved multi
task navigation datasets and directly learning navigation skills,
showing initial success in multi-task performance. However,
these methods experienced performance drops when deployed
in novel environments, especially in real-world settings. In
recent years, advanced approaches [114, 61, 35, 60, 121, 78]
have leveraged the generalization capabilities of large language
models to improve multi-task navigation. These models show
promising generalizability across navigation tasks but rely on
extensive prompting, which impacts time efficiency. In contrast,
our video-based large language model is trained end-to-end
for multi-task navigation, offering robust generalization and
for data collection and evaluation. Finally, we collect 3.6M
computational efficiency for tasks like human following.
evaluate our method on mainstream datasets to clearly justify
the performance of our approach.
are consistently being proposed, covering a diverse range
of navigation attributes. However, our goal is to train and
benchmark using Habitat 3.0 [68]. Note that new benchmarks
and MX-EQA [38]. We select MP3D-EQA, which is well
maintained with the latest baselines. For human-following
[116, 117] benchmarks, there is currently no benchmark
that provides textual descriptions of humans. Therefore, we
have self-built a textual description-based human-following
natural language instruction I consisting of l words and an
ego-centric RGB video OT comprising a sequence of frames
{x1,··· ,xT}, the agent is required to plan the next k actions
{AT,··· ,AT+k−1} to executed for complete the instruction
within novel environments (k = 4 in our experiments). Here,
we adopt a widely used action setting [76, 12, 44, 21], which
require the agent to take low-level actions a ∈ A, including
navigation of Uni-NaVid as follows: At the time T, given a
Navigation task definition. We define the general-purpose
such as MP3D-EQA [90], MT-EQA [103], Graph-EQA [83],
III. PROBLEM FORMULATION
are diverse datasets focusing on different attributes of EQA,
execution in real-world environments.
as VLN-CE. For embodied question answering (EQA), there
efficiency for long-horizon tasks and supporting non-blocking
dataset on Habitat [76], which shares the same action settings
environments and simulators. Here, we leverages the HM3D
in long-horizon tasks. In contrast, Uni-NaVid implements
an online visual token merging strategy, optimizing training
MP3D [11], and Aithor [124], which are built on various scene
continuous movement. However, it faces efficiency challenges
there are several famous benchmarks such as HM3D [69],
language model end-to-end with low-level actions to enable
continuous environments, called VLN-CE [44], which is more
practical for real-world applications. For object goal navigation,
ments. Another approach [111, 106] trains a video-based large
provide navigation instructions and ground truth trajectories
of landmarks. We focus on a variant of R2R and RxR in
into text and relying on discrete landmarks results in sparse
environmental observations and is limited to static environ
2-Room (R2R) [4] and Room-cross-Room (RxR) [47], which
format, prompting the language model to select landmarks that
guide the agent. However, abstracting dense visual information
language navigation, the most widely used datasets are Room
models [22, 54] to describe surrounding environments in text
the datasets most relevant to our methods. For vision-and
zero-shot manner. These methods employ visual foundation
role in the embodied navigation community. Here, we review
61, 60, 78] is to use off-the-shelf large language models in a
been proposed [23, 123, 59, 64]. These datasets play a crucial
understanding and planning. One straightforward approach[120,
Embodied Navigation Datasets. To train and evaluate the
performance of a policy for embodied navigation tasks, a
large body of datasets and corresponding benchmarks have
Models (LLMs)[20, 54, 122] have been introduced into
robotic navigation due to their generalization capabilities in
Large Language Models for Navigation. Large Language
Fig. 2: Pipeline of Uni-NaVid. Our method takes only single-view RGB frames {x1,··· ,xT} and a natural language instruction
I as input. For each frame, we extract 64 visual tokens using the vision encoder and then use online token merging to accelerate
the model while retaining compact visual information. The merged tokens and instruction tokens are sent to the large language
model to obtain actions for navigation or answers for embodied question-answering.
if
( (
:
{                  }
,
else
:
The    plant    is   green 
Language Token
Image Token
Grid Pooling
Nav indicator
Large Language Model
1
Long-Term Memory
Short-Term Memory
Current Observation
Language
M×1
&
Grid pooling
Merging
Online Visual Token Merging 
B×4
Grid pooling
1×64
Vision Encoder
T-B-1
T-1
1×1 2×2 8×8
(2) Question
What is the color of the plant?
x1 x2 xT-B-1 xT-B xT-2 xT-1 xT
T
History Video
(1) Instruction
Walk forward to the 
door then turn left, and
move to the plant and 
stop.
VQA
E-QA
Follow
ObjNav
VLN
{FORWARD,TURN-LEFT,TURN-RIGHT,STOP}. Note that,
are grouped based on their timestamps relative to the current
our task formulation is compatible with existing embodied
navigation tasks [76, 12, 44, 21], where the discrete low-level
actions [76, 12, 44, 21] represent a small rotation (30 degrees)
or a forward movement (25 cm), making them flexible to be
used in continuous environments such obstacle avoidance. We
provide a detailed explanation of how these actions are applied
in both synthetic and real-world environments in Sec. VI-A
Overview. As illustrated in Figure 2, Uni-NaVid is com
posed of three main components: a vision encoder, an online
token merge mechanism and a large language model (LLM).
First, the online captured video stream is encoded by the vision
encoder (EVA-CLIP [82] in implementation) to extract frame
wise visual features in the form of tokens, which we denote
them as visual tokens. The visual tokens are then spatially
and temporally merged by leveraging an online token merge
mechanism. Next, the merged visual tokens are projected
with an MLP projector into a feature space aligned with
language tokens, which are referred to as visual observation
tokens. As common, the instructions are also tokenized as a
set of tokens, known as language observation tokens. Both the
visual observation tokens and language observation tokens are
concatenated and passed to the Large Language Model (LLM),
which infers four action tokens that represent the next four
actions.
IV. MODEL OF UNI-NAVID
A. Observation Encoding.
Given the ego-centric video up to time T, denoted by OT =
{x1.··· ,xT}, we encode the video to a sequence of visual
features in the form of tokens. For each frame xt, we first get
its visual feature tokens Xt ∈ RNx× C with a vision encoder
(EVA-CLIP [82] in implementation), where Nx is the patch
number (Nx is set to 256) and C is the embedding dimension.
X1:T = Encoder(x1:T)
(1)
The visual features provide rich information that enables the
agent to understand its navigation history and plan subsequent
actions. However, during navigation, the progressively increas
ing number of visual tokens (T ×Nx) results in progressively
longer inference times for the LLM (typically 1–2 seconds
per inference) [111]. This increased latency renders LLM
based navigation impractical for deployment in real-world
environments.
B. Online Visual Token Merging
To reduce the number of visual tokens while preserving
sufficient navigation visual information, we design an token
merging mechanism. This strategy is based on the key insight
that recent observations are more critical for navigation, and
that visual information between consecutive frames (temporally)
and within neighboring pixels (spatially) may be redundant.
Visual token grouping. Drawing inspiration from the
Atkinson-Shiffrin memory model [6, 80], we categorize visual
tokens into current visual tokens Xcurr, short-term visual tokens
Xshort, and long-term visual tokens Xlong. These visual tokens
frame T and for each group of visual tokens, we apply a grid
pooling operation at different pooling resolutions:



X1:T =


Xcurr = GridPool(Xt,αcurr),
if t = T
Xshort = GridPool(Xt,αshort), if t ∈ [T-B, T)
Xlong = GridPool(Xt,αlong), if t ∈ [1, T-B)
(2)
where GridPool(·) is a grid pooling operation [51, 111],
spatially squeezing the tokens from Nx to Nx
α2 
, and B (set to 64)
is the length of the buffer of shorter memory. Here, we adopt
the αcurr = 2, αshort = 8, αlong = 16, leads to visual tokens as
Xcurr ∈ R64× C, Xshort ∈ R4× C, Xlong ∈ R1× C, respectively.
Here, current visual tokens Xcurr encapsulate comprehensive
visual information, enabling the agent to perceive its immediate
environment and plan subsequent trajectories. Meanwhile,
Xshort and Xlong capture temporally rich information from the
captured video stream, facilitating the agent’s comprehension
of its navigation history.
It should be noted that these hyperparameters are obtained
through empirical experimentation to achieve an optimal
balance between manageable token numbers and adequate
visual information representation. These hyperparameters can
be further adjusted when memory capacity and computational
resources are not limiting factors. We provide a detailed
explanation and ablation study of α in the supplemental
material.
Online visual token process. During the navigation pro
cess, the agent consistently observes new frames. However,
performing encoding and grouping (Eq. 2) for all frames at
each step would be computationally intensive. To address this,
we implement an online visual token processing mechanism
that maximizes the reuse of previously generated visual tokens.
Specifically, when a new frame at time T + 1 is received, we
apply grid pooling exclusively to the most recent visual tokens
at time T and the oldest short-term visual tokens at time T −B.
These processed tokens are then integrated into the short-term
and long-term visual tokens, respectively:
Xcurr→short = GridPool(Xcurr, αshort
αcurr 
),
Xshort→long = GridPool(Xshort, αlong
αshort 
).
(3)
(4)
To prevent the linear growth of long-term visual tokens
XLong, we further perform token merging on the long-term
visual tokens by combining adjacent tokens that exhibit high
similarity, following the approach of VLM-based methods [8,
80]. Specifically, we merge the long-term visual tokens based
on the cosine similarity between Xshort→long and the most recent
long-term visual tokens Xlong at time T−B−1. If the similarity
exceeds a predefined threshold τ, we merge them according
to the number of frames previously merged (denoted as K) in
the latest long-term visual tokens:
when handling longer video sequences. A detailed analysis of
Algorithm 1 Online Visual Token Merging
Require:
• Total number of frames T
• Short memory buffer length B
• Grid pooling scales: αcurr, αshort, αlong
• Current visual tokens: XT ∈ RNx× C
• Previously merged tokens: Xcurr, Xshort, Xlong
• Number of frames merged in the last tokens of long
memory: K
Ensure:
• Updated merged tokens: X′
curr, X′
short, X′
long
• Updated number of frames merged in the last tokens of
long memory: K′
1: if T == 1 then
2:
X′
short, X′
long ← []
3: else
4:
▷ First frame, empty history tokens
▷ Update short-term visual tokens
Xcurr→short ← GridPool(Xcurr, αshort
5:
X′
short ← Xshort + [Xcurr→short]
6: end if
αcurr 
)
7: X′
curr ← GridPool(XT,αcurr) ▷ New current visual token
8: if T > B+1 then ▷ Out of short-term tokens buffer
9:
10:
11:
12:
13:
14:
15:
16:
17:
18:
19:
Xshort→long ← GridPool(Xshort[0], αlong
αshort 
)
X′
short ← Xshort[1 :]
s ←cos(Xlong[−1],Xshort→long)
if T > B+2 and s>τ then ▷ Fuse long-term tokens
Xlast long ← 1
K+1(KXlong[−1] + Xshort→long)
X′
long ← Xlong[: −1] + [Xlast long]
K′ ←K+1
else
▷ Add new long-term token
X′
long ← Xlong + [Xshort→long]
K′ ←1
end if
20: end if
Xlong = 1
K+1 (KXlong +Xshort→long),
subject to cos(Xlong,Xshort→long) > τ.
(5)
(6)
We insert new long-term visual tokens Xshort→long when
their similarity falls below a threshold τ (empirically set
to τ = 0.95 [80]), indicating that they contain relatively
distinct visual information. This online visual token processing
preserves the navigation visual history in a highly compact
form (with a length of M ≪ T − B − 1). Notably, only
visual tokens at the boundaries of groups require parallelizable
grid pooling, making the process computationally efficient and
naturally suited for online deployment in real-world navigation
tasks. We give a description of our token merging technique
at Algorithmn 1.
Compared to existing video-based large language mod
els [111, 80, 51], this online merging strategy significantly
reduces inference time, achieving an average of 0.2 seconds
time efficiency is provided in the Supplementary Materials.
C. Action Planning
After obtaining the merged visual tokens from semantic
features [82], we adopt established practices in Vision-and
Language models [54, 51] to perform vision-language align
ment, enabling the large language model (LLM) to effectively
interpret visual information. Specifically, we leverage a cross
modality projector PV (·) to project all merged visual tokens
Xmerged = {Xlong,Xshort,Xcurr} into visual observation tokens
that are compatible with the LLM’s input representation space:
EV
T = PV(Xmerged),
(7)
where the PV (·) is implemented as a two-layter MLP [54]
and optimized in an end-to-end training manner. For instruction
encoding, we use the off-the-shelf language tokenizer and
embeing layer of LLM (Vicuna-7B [20]) to encode navigation
instruction into language observation tokens EL
T. Then we
concatenate the visual observation tokens EV
T , a navigation task
indicator ⟨NAV⟩ and language observation tokens EV
T form the
f
inal input token sequence. Here, the navigation task indicator
⟨NAV⟩ is adopted by following [111, 67] for accelerating the
specific task learning and obtaining consistent output format.
Finally, the complete input token sequence is fed into the LLM
to infer four action tokens {EA
T,··· ,EA
T+3}, as described
below. We include a discussion on the input token format in
the Supplementary Material
Input: {Long term tokens}{Shot term tokens}
{Current tokens} <NAV > {Instruction}
Output: <Action 0><Action 1><Action 2>
<Action 3>
The action tokens belong to the discrete action set
{FORWARD,TURN-LEFT,TURN-RIGHT,STOP}. Following
the standard configuration in existing navigation settings [76,
106], the forward action corresponds to a movement of 25
cm, and the turning actions represent a 30◦ rotation. This
configuration is consistent with all training navigation data
(Sec. V). Empirically, we find that predicting the next four steps
yields optimal performance, which encourages Uni-NaVid to
forecast long-horizon action sequences while still considering
sufficient observations for accurate prediction. This multi-step
prediction also supports asynchronous deployment, enabling
non-blocking navigation performance in the real world. Please
see the Supplementary Material for detailed elaboration.
V. DATA COLLECTION AND TRAINING
To train Uni-NaVidfor mastering multi-navigation tasks,
it is crucial to gather extensive and diverse navigation data
across various tasks and environments. However, directly
collecting large amounts of real-world navigation data can be
prohibitively expensive. To address this challenge, we propose
per inference. This improvement becomes increasingly notable
two key strategies for training Uni-NaVid: First, we collect
Continue
Bathroom
Couch Kitchen
Find
Exit
Advance Endpoint
Mirror
Left
Destination
Glass
Sofa
Proceed
Number of Frames
Corner
Reach
Walk
Hallway
Entrance
Move
See
Front
Wall
Facing
Step
Follow
Window
Restroom
Turn
Table
Room
Bedroom
Passing
Pivot
Right
Door
Staircase
VLN
40%
Standing
Chairs
Enter
Area
Corridor
Embodied
Navigation
61%
Number of Samples
VideoQA
9%
Open-world
Video
39%
ObjectNav
8%
EQA
4%
Following
9%
Women
hite t-shirt
Blue jeans
Fig. 3: Visualization of training data. We visualize the
Blue jeans
combination of training data (5.9M), video frame counts, and
the most common words in navigation instructions.
multi-task navigation data from a wide range of synthetic
environments (totaling 861 scenes) using a uniform input
and output format, enabling Uni-NaVidto acquire general
navigation skills. Second, we co-tune Uni-NaVidwith real-world
video-based question-answering data, enhancing its ability to
interpret real-world images and supporting its open-vocabulary
knowledge acquisition.
A. Multi-Task Navigation Data.
We collect the largest multi-task navigation dataset to date
within the Habitat simulator environment [76], comprising
3.6 million samples across four distinct navigation tasks,
as described below. All tasks are curated within a unified
framework. A detailed data collection strategy is provided in
the Supplementary Materials.
(A) Vision-and-language navigation [44, 46] require the
agent to interpret and ground instructions in visual observations,
effectively combining linguistic and visual information to make
sequential decisions. Specifically, the agent has to navigate
based on landmarks and motions described in the text and stop
nearby the correct destination. Here, we collect 2.4M navigation
samples of mainstream VLN datasets, VLN-CE R2R [44] and
RxR [47], that focus on continuous environments.
(B) Object Goal Navigation [76] involves an agent nav
igating an environment to locate a specific object based on
provided visual or linguistic cues. This task evaluates the
agent’s ability to perceive objects, understand scene layout, and
execute efficient search strategies. We collected 483k samples
from datasets in the Habitat Matterport 3D dataset (HM3D
ObjectNav) [70]. Note that, in HM3D ObjectNav, the agent is
required to locate objects from a predefined category set (e.g.,
sofa, chair, and bed). Nevertheless, experiments demonstrate
that our method generalizes to SOTA-level open-vocabulary
object goal searching, as shown in Table V.
(C) Embodied question answering [90] requires the agent to
navigate to the related area for question answering. It involves
Video
Caption
30%
Man
Women
White t-shirt
Blue jeans
…
Light-green Top
Denim shorts
…
Man
Blue t-shirt
Blue jeans
…
Instruction example:
“Follow the woman in a 
light gray t-shirt and blue 
jeans.”
Man
Blue
Blue jeans
Fig. 4: Language-described human following benchmark.
Women
White t-shirt
Blue jeans
We construct our human-following benchmark based on Habitat
3.0 [68] by incorporating textual descriptions for each avatar
(eight in total, top row). The robot is required to comprehend
these descriptions and accurately follow the designated indi
vidual in crowded environments.
spatial reasoning, object description, and understanding contex
tual information, requiring the ability to integrate perception,
language comprehension, and decision-making. Following the
setup in main stream EQA methods [21, 90], the agent first
navigates to the target related to the question, issues a stop
action, and then provides an answer. We collect 240k video
action samples and 10k video-answering samples on the MP3D
EQA dataset [21] on Matterport 3D environments [11].
(D) Human following [37, 25] requires the agent to track and
follow a human target with a specific description in dynamic
and crowded environments, e.g., “Follow the man in the blue t
shirt.”. The agent must recognize the appearance of the human,
follow the correct person described in the instructions, predict
their movement trajectory, and keep an appropriate distance
while avoiding obstacles.
However, there is currently no human-following dataset
that supports language-described human following in crowded
environments (multi-person scenarios). To this end, we extend
the Habitat 3.0 social navigation benchmark [68] by (1) adding
textual descriptions for each avatar (8 in total, as illustrated in
Fig. 4), (2) introducing additional distracting human avatars
to simulate challenging real-world environments, and (3)
deploying the robot and humans in the Habitat Matterport
3D dataset [101], which offers photo-realistic rendering quality
and diverse large-scale scenes. The robot and target human
are initialized nearby (using the same setting as [68]), with
randomly moving distracting human avatars. Based on this
setup, we collected 544k human-following navigation samples.
We also add a detailed description in Supplementary Material.
This benchmark will also be released to benefit the navigation
community.
Unified navigation samples. The data statistics are presented
in Figure 3. It is worth noting that the number of samples
in VLN is relatively larger compared to other tasks. This
is because VLN [44, 46] requires the agent to navigate all
landmarks described in the instructions, which often results
in longer trajectories and, consequently more video-action
samples. Here, we collect all navigation samples in a uniform
format, including an egocentric RGB video, a natural language
instruction, and four corresponding future actions. All data were
collected from synthetic scenes across the Habitat-Matterport
3D (HM3D) and Matterport 3D (MP3D) datasets. We use
the default settings of each environment, with a height range
of 0.88 m to 1.25 m and a robot radius between 0.1 m and
0.6 m. This approach helps prevent overfitting to a specific
robot embodiment. This approach helps prevent overfitting
to a specific robot embodiment. Note that while there exist
insightful techniques [24, 29] investigating navigation for robots
of general sizes, our focus is primarily on uniform multi-task
navigation.
B. Training Strategy of Uni-NaVid
Joint training on synthetic and real-world data. Although
we collect navigation data from various environments, the
diversity in both observations and instructions remains limited
to a specific set of synthetic environments. To incorporate open
world knowledge, we follow previous Vision-and-Language
Action models [111, 9], integrating open-world video question
answering during training. Specifically, we adopt a two-stage
training process (a common strategy in Vision-and-Language
models [54, 51, 80]): (1) First, we exclusively train the cross
modality projector (Equ. 7) using the same modality alignment
dataset as LLaMA-VID [51]. (2) Second, we fine-tune both
the projector and the Large Language Model (LLM) using
2.3M video question-answering data from publicly available
datasets [7, 19, 51], along with 3.6M multi-task navigation
samples. During training, we apply the online token merging
to both the VQA samples and navigation samples, the only
difference is the VAQ samples do not include navigation task
indicator ⟨NAV⟩.
Training configuration. Uni-NaVid is trained on a cluster
server with 40 NVIDIA H800 GPUs for approximately 35
hours, totaling 1400 GPU hours. For video data, we sample
frames at 1 FPS to remove redundant information between
consecutive frames. During training, the vision encoder (EVA
CLIP [82]) and large language model (Vicuna-7B [20]) are pre
loaded with default pre-trained weight. Following the training
strategy of VLM [54], we optimize the trainable parameters
for only 1 epoch.
VI. EXPERIMENT
We conduct experiments to evaluate Uni-NaVid on three
specific aspects: (1) How does Uni-NaVid perform on individual
tasks? (2) Does learning multiple navigation tasks lead to
synergistic improvements? (3) Is the key design of our method
effective? To evaluate the general-purpose navigation method,
we conduct extensive experiments on individual navigation
tasks, employing corresponding strong baselines. Additional
details are provided in the supplemental material.
Benchmarks. We evaluate our method on various bench
marks across different navigation tasks. Given the diversity of
benchmarks spanning various environments and simulators, we
meticulously verify the scene splits to ensure no overlap exists
between the training and validation scenes across benchmarks.
• Vision-and-language navigation: We test our method
on the validation splits of the VLN-CE R2R [44] and
RxR [46] benchmarks.
• Object goal navigation: We use the validation split
of the Habitat Matterport 3D (HM3D) dataset [70],
which requires the agent to find target objects from six
categories (sofa, chair, TV, bed, toilet, and plant) in unseen
environments. Moreover, to test generalizability, we also
evaluate our method on the HM3D-OVON dataset [101],
an open-vocabulary object navigation benchmark, in a
zero-shot manner.
• Embodied question-answering: We use the validation
split of the MP3D-EQA benchmark [90]. Additionally,
we conduct experiments on the more recent Embodied
Video Question Answering benchmark, OpenEQA [63].
• Human following: We evaluate our method along
side mainstream approaches on our proposed language
described human following benchmark.
• Video understanding: We follow the evaluation proce
dures of existing VQA methods [51]. We choose the
ScanQA [7], MSVD [13], MSRVTT [96], and Activi
tyNet [10] datasets.
Metrics. To evaluate navigation performance, we follow the
standard evaluation metrics [4], including success rate (SR),
oracle success rate (OS), success weighted by path length
(SPL) [3], trajectory length (TL), following rate (FR) [68],
collision rate (CR) [68] and navigation error from goal (NE).
Note that the success criteria change among different navigation
tasks, we therefore use the default success criteria of each
benchmark. For video understanding evaluation, we employ
widely used metrics following existing works [7, 51].
A. Deployment Details of Uni-Navid.
Benchmark evaluation. For each navigation task, we adhere
to the default settings of each navigation task [44, 76, 21, 37].
All tasks take an online captured RGB video (capturing one
frame after each action) and a textual instruction as inputs,
and output the next four actions (Sec. IV-C). The robot then
executes the predicted actions and calls STOP once the first
predicted action is a stop action. For VLN and EQA tasks, we
directly use the text instruction provided by the benchmark
episodes. For human following and object goal navigation, we
transform the target information into an instruction by adding
prefixes such as ”Search for” or ”Follow.” Further details can
be found in the supplemental material.
It is worth noting that for EQA [21] task, the agent executes
navigation actions until a stop command is issued. We then
remove the navigation-specific token <NAV> and query the
questions using the navigation history. This strategy alleviates
the ambiguity for the LLM in deciding whether to navigate or
answer a question (See Table X).
Real-world deployment. For real-world deployment, we
utilize a remote server with an NVIDIA A100 GPU to run
Method Observation VLN-CER2RVal-Unseen
Pan.Odom.DepthS.RGB TL NE↓OS↑SR↑SPL↑
HPN+DN∗ [45] ✓ ✓ ✓ 7.62 6.31 40.036.0 34.0
CMA∗ [31] ✓ ✓ ✓ 10.906.20 52.041.0 36.0
VLN
⟳BERT∗†[31] ✓ ✓ ✓ 12.235.74 53.044.0 39.0
Sim2Sim∗ [43] ✓ ✓ ✓ 10.696.07 52.043.0 36.0
GridMM∗ [87] ✓ ✓ ✓ 13.365.11 61.049.0 41.0
HAMT∗‡[89] ✓ ✓ ✓– 4.80– 55.0 51.0
ETPNav∗ [1] ✓ ✓ ✓ 11.994.71 65.057.0 49.0
InstructNav[61] ✓ ✓ ✓ ✓ 7.74 6.89- 31.0 24.0
AG-CMTP[15] ✓ ✓ ✓– 7.90 39.223.1 19.1
R2R-CMTP[15] ✓ ✓ ✓– 7.90 38.026.4 22.7
LAW[73] ✓ ✓ ✓ 8.89 6.83 44.035.0 31.0
CM2[26] ✓ ✓ ✓ 11.547.02 41.534.3 27.6
WS-MGMap[16] ✓ ✓ ✓ 10.006.28 47.638.9 34.3
ETPNav.FF[88] ✓ ✓ ✓- 5.95 55.844.9 30.4
Seq2Seq[44] ✓ ✓ 9.30 7.77 37.025.0 22.0
CMA[44] ✓ ✓ 8.64 7.37 40.032.0 30.0
NaVid[111] ✓ 7.63 5.47 49.137.4 35.9
Uni-NaVid ✓ 9.71 5.58 53.347.0 42.7
TABLEII:Vision-and-language navigation (R2R). Com
parisononVLN-CER2R[44]Val-Unseen. ∗:Methodsuse
high-level actionspace. †:Methodsuse thesamewaypoint
predictorproposedin[31]. ‡:Methodsuseadditionalvisual
datathanMP3Dscenes[11].
Method Observation VLN-CERxRVal-Unseen
Odom.DepthS.RGB TL NE↓OS↑SR↑SPL↑
LAW*[73] ✓ ✓ ✓ 4.01 10.8721.0 8.0 8.0
CM2*[26] ✓ ✓ ✓ 12.29 8.98 25.3 14.4 9.2
WS-MGMap*[16] ✓ ✓ ✓ 10.80 9.83 29.8 15.0 12.1
ETPNav.FF[88] ✓ ✓ ✓- 8.79 36.7 25.5 18.1
Seq2Seq*[44] ✓ ✓ 1.16 11.8 5.02 3.51 3.43
CMA*[44] ✓ ✓ 5.09 11.7 10.7 4.41 2.47
A2Nav† [17] ✓––– 16.8 6.3
NaVid*[111] ✓ 10.59 8.41 34.5 23.8 21.2
Uni-NaVid ✓ 15.8 6.24 55.5 48.7 40.9
TABLEIII:Vision-and-languagenavigation(RxR).Com
parisononVLN-CERxR[47]Val-Unseen. ∗:onlytrainedon
VLN-CER2R.
Uni-NaVid,whichprocesses observations (alongwith text
instructions)andsendscommands toalocal robot toexecute
thepredictedactions.Uni-NaVidrequiresapproximately0.2
seconds togeneratethenext fouractions.Duringnavigation,
therobot asynchronouslycompressesanduploads the latest
observations to themodelwhileexecutingpendingactions.
Refer to thesupplementaryvideofor real-worldnavigation
performance.
B. IndividualTaskResults
Comparisononvision-and-languagenavigation.Weeval
uateourmethodwithmainstreambaselinesontwopublicly
availablebenchmarks:VLN-CER2R[44]andRxR[47].The
resultsareshowninTableIIandTableIII.Wefindthatour
methods achieveSOTA-level performanceonbothdatasets
usingonlyRGBvideos as observations. Incomparison to
NaVid [111], which is also a vision languagemodel that
is solely trainedonVLNdata, our approachdemonstrates
significant improvements,witha+25.7%increaseinSuccess
Rate(SR)onR2R.Forzero-shotmethods(InstructNav[61]
Method Observation HM3DObjectNav
Odom. Depth S.RGB SR↑ SPL↑
DD-PPO[91] ✓ ✓ ✓ 27.9 14.2
Habitat-Web[71] ✓ ✓ ✓ 57.6 23.8
InstructNav[61] ✓ ✓ ✓ 58.0 20.9
PIRLNav-IL[72] ✓ ✓ 64.1 27.1
PIRLNav-IL-RL[72] ✓ ✓ 70.4 34.1
OVRL[99] ✓ ✓ 62.0 26.8
OVRL-v2[98] ✓ ✓ 64.7 28.1
Uni-NaVid ✓ 73.7 37.1
TABLEIV:Objectgoalnavigation.ComparisononHabitat
Matterport3D[70]ObjectNavdataset.
Method VALSEEN VALSEEN
SYNONYMS VALUNSEEN
SR↑ SPL↑ SR↑ SPL↑ SR↑ SPL↑
BC 11.1 4.5 9.9 3.8 5.4 1.9
DAgger 11.1 4.5 9.9 3.8 5.4 1.9
RL 18.1 9.4 15.0 7.4 10.2 4.7
BCRL 39.2 18.7 27.8 11.7 18.6 7.5
DAgRL 41.3 21.2 29.4 14.4 18.3 7.9
VLFM∗ [100] 35.2 18.6 32.4 17.3 35.2 19.6
DAgRL+OD[101] 38.5 21.1 39.0 21.4 37.1 19.8
Uni-NaVid∗ 41.3 21.1 43.9 21.8 39.5 19.8
TABLEV:Objectgoalnavigation.ComparisononHM3D
OVON[101]. ∗ :denoteszero-shotmethods.
andA2Nav [17]) that useChatGPTwithonly text inputs
forvisual languagenavigation(VLN), theseapproachesoften
face challenges in transitioningbetween text prompts and
visual information, resultinginless thansatisfactoryoutcomes.
Furthermore, it is important tonote that the trajectories in
RxRaremorediverseandinvolvelongerpathswithdetailed
landmarkdescriptions,makingRxRwidelyregardedasmore
challengingthanR2R.However,ourmethodachievesconsistent
performanceacrossbothR2RandRxR,withslightlybetter
results onRxR(+3.6SR(%)), demonstrating its ability to
effectivelyleveragedetailedinstructions tonavigatediverse
trajectories.Weaddexperimentsof removingRxRsamples in
SupplemntalMaterial,whereourmethodstill achiveSTOA
performance(+23.9SR(%))againstNaVid.
Comparisononobjectgoalnavigation.Weconduct the
experiments onHM3D [70] to compareUni-NaVidwith
mainstreammethods[91,71,72,99,98] thatalsolearnfrom
ObjectNavdata.Theresults, showninTableIV,demonstrate
that our approachachieves thebest performance.Note that
methodsnotutilizingodometryfacechallengesas theymust
relyon implicitmemory to retain the historical trajectory.
Nevertheless,Uni-NaVidstillachievessignificantgains inSR
(+4.7%)andSPL(+8.8%)comparedtopreviousstate-of-the
artmethods.Additionally,webelieveourmethod’sObjectNav
performancecanbefurtherenhancedbyincorporatingreinforce
ment learningtechniques, asdemonstratedbyPIRLNav[72]
andPoliformer [106].
Toevaluatethegeneralizationabilityforopen-vocabulary
objects,weevaluateourmethodontheopen-vocabularyobject
goalnavigationbenchmark(HM3D-OVON[101]) inazero
Method ActionType MP3DEQA
D.E.C.E.GT ACC↑
NaviLLM[114] ✓ 44.5
Uni-NaVid ✓ 47.3
EQA(habitat-lab) [21] ✓ 46.0
NaviLLM[114] ✓ 47.4
Uni-NaVid ✓ 54.4
TABLEVI:Embodiedquestionanswering.Comparisonon
HabitatMatterport3DEQAdataset [21].
Method Observation HumanFollowingDataset
H.Det. S.RGB SR↑ FR↑ CR↓
PoliFormer [106] ✓ 2.79 20.35 2.93
PoliFormer∗ [106] ✓ ✓ 14.67 37.14 4.29
PoliFormer†[106] ✓ ✓ 25.29 47.16 6.78
IBVS∗ [28] ✓ ✓ 46.08 62.64 0.84
IBVS†[28] ✓ ✓ 50.58 68.89 0.80
Uni-NaVid ✓ 61.21 71.93 2.07
TABLEVII: Human following. Comparison on Human
FollowingDataset.∗:MethodsuseGroundingDINO[58]asthe
open-vocabularyhumandetector. †:Methodsusetheground
truthboundingboxprovidedbythesimulator.
shotmanner. The results inTableVdemonstrate that our
methodachievessignificant improvementover thezero-shot
method(VLFM[100])andevenoutperforms thefine-tuned
method (DAgRL+OD[101]) on theVALSEENandVAL
UNSEENsplits.Thisprovesthegeneralizabilityofourmethod.
Comparisononembodiedquestionanswering.Theevalu
ationresultsonMP3D-EQA[90]arepresentedinTableVI.
Despite navigating in continuous environments (CE), our
methodoutperformsexistingapproaches(e.g.,NaviLLM[114]
leveragethesameevaluationstrategyinSec.VI-A)thatoperate
withindiscretelandmark-basedenvironments(DE).Moreover,
whenprovidedwiththegroundtruth(GT)navigationtrajectory,
ourmethodshowsasignificant improvement,demonstrating
itsabilitytounderstandnavigationhistoryeffectively.Wealso
report our performanceon themore challengingEM-EQA
benchmark,OpenEQA[63], in theSupplementalMaterial,
whichincludesmorecomplexquestions.Ourmethodachieves
comparableperformancetoGPT-4Vwithscenecaptions[63].
Comparison on human following.We compared our
methodwith twomost relativemethods PoliFormer [106]
andIBVS[28].Sincebothmethodsrequireaspecifichuman
boundingboxas input,obtainedfromanupstreamalgorithm,
weusetheboundingboxfromtheopen-worldobjectdetector
GroundingDINO[58]andthegroundtruthprovidedbythe
simulator toevaluate thehuman followingperformanceof
thecomparisonmethodsundervarioussetups.Asshownin
TableVII,Uni-NaVidoutperforms thecomparisonmethods
onbothSR(+21.0%) andFR(+4.4%)whilemaintaining
lowCRunder anysetup, evenwhen theyuseground truth
boundingboxesasinput.ThisdemonstratesthatUni-NaVidcan
effectivelyinfer instructionsandfollowthecorrecthuman, as
Method ScanQA
EM↑BLUE-1↑ROUGE↑METEOR↑CIDEr↑
V.N.+MCAN[105] 19.71 29.46 30.97 12.07 58.23
S.R.+MCAN[105] 20.56 27.85 30.68 11.97 57.56
3D-LLM(flamingo) [32] 23.2 32.60 34.80 13.5 65.6
NaviLLM[114] 26.27 39.73 40.23 16.56 80.77
BridgeQA[65] 31.29 34.49 43.26 16.51 83.75
Uni-NaVid 28.01 46.85 45.74 19.24 94.72
TABLEVIII:Embodiedvideoquestionanswering.Compar
isononScanQA[7]benchmark.
Method MSVD-QA MSRVTT-QAActivityNet-QA
Acc↑Score↑Acc↑ Score↑ Acc↑ Score↑
VideoLLaMA[107] 51.6 2.5 29.6 1.8 12.4 1.1
VideoChat [49] 56.3 2.8 45.0 2.5 26.5 2.2
VideoChatGPT[62] 64.9 3.3 49.3 2.8 35.2 2.7
BT-Adapter [56] 67.5 3.7 57.0 3.2 45.7 3.2
Chat-UniVi [39] 65.0 3.6 54.6 3.1 45.8 3.2
LLaMA-VID[51] 69.7 3.7 57.7 3.2 47.4 3.3
VideoChat2[50] 70.0 3.9 54.1 3.3 49.1 3.3
Video-LLaVA[53] 70.7 3.9 59.2 3.5 45.3 3.3
ST-LLM[57] 74.6 3.9 63.2 3.4 50.9 3.3
Uni-NaVid 69.6 3.9 59.3 3.5 51.4 3.7
TABLEIX:Videoquestionanswering. Comparisonwith
leadingmethods (all based onVicuna-7B [20]) onVQA
benchamarks.
wellaspredict thehuman’smovementpatternsaccurately.We
include additional human-followingexperiments invarious
environments, suchasHSSD[42] andMP3D[11], in the
SupplementalMaterial.Ourmethodconsistentlydemonstrates
SOTAperformanceacross thesesettings.
Comparisononvideoquestionanswering.Wefirstevaluate
our method on ScanQA [7] on Tab. VIII. Compared to
mainstreambaselines,wefindthatUni-NaVidarchivesthebest
performanceonfourmetrics, includingBLUE-1(+17.9%),
ROUGE(+5.7%),METEOR(+16.2%),andCIDEr(+13.1%).
Thisproves thesuperiorityofourmethodsonspatial scene
understanding.NotethattheEMmetricrequiresanexactmatch
betweenthequestionandanswer,whichisnotwell-suitedto
ourmethod, as it isdesignedtolearnfromdiversedataand
generateflexibleresponses.
We further evaluate our method on open-ended video
question-answering benchmarks [13, 96, 10], as presented
in Table IX. To ensure a fair comparison, we focus on
methodsthatemploythesamelargelanguagemodelbackbone
(Vicuna-7B[20]).Theresultsindicatethatevenafterextensive
tokenmerging(Sec. IV-B),Uni-NaVidachievesperformance
comparabletostate-of-the-artmethods.Thisdemonstrates the
effectivenessofbothour tokenmergingand trainingstrate
gies,whilealsohighlightingrobustopen-worldunderstanding
capabilities.
C. QualitativeResults inReal-World
Weconductedextensiveexperimentsonreal-worldenviron
ments (experiment detailsareprovided in thesupplemental
What is the color of the trash bin? Search for a doll.
Third-person view Trajectory
Follow the man with blue sweater.
Move forward and turn left. Then move to the glass door and turn 
left. And walk along the hallway, turn right when you near the 
plant. Then move forward and stop in front of the double door.
Go to the sofa, then turn left and move into the door, thenturn
left and stop by the person on a chair.
Third-person view Trajectory
(A) Vision-and-Language Navigation (B) Object Goal Navigation (C) Embodied Question Answering (D) Human Following
Find a chair with a doll. What is the color of the flower 
close to the TV?
: Blue
Follow the person.
: Black
Fig.5:Visual resultsofUni-NaVidinreal-world.WedeployUni-NaVidacrossdiverseenvironments toexecuteinstructions
inazero-shot setting.Weprovidethird-personviewswithrobot’s trajectory, showingeffectivenavigationperformance.We
indicatethestartingpointasabluedotandtheendingpointasagreenarrow.
Trajectory (A)
Moveforwardtodoorthenenterthedoor. Searchforapersonand
followtheperson. How many chairs are in the room?
Start
End
: One
1 2
3
Video
… …
Steps
1 2 3
Trajectory (B)
Movetothemanontherightside.
asofa.Turnrightand search for a TV, stop by the TV. Finally, what 
is the color of the sofa?
Start
1
2
3
Video
…
1 2 3
End
… … …
Movetothemanontherightside.Then followthatmanuntilyou
seeasofa.Turnrightand search for a TV, stop by the TV. 
Finally, what is the color of the sofa?
Uni-NaVid: Beige
: Beige
2
3
Steps
Fig. 6: Vivusal results of Uni-NaVid on compostional
tasks.Theagent isrequiredtoexecutecomplexinstructions
involvingmultiplenavigationtasks.Ourmethodsuccessfully
accomplishesthesenavigationtaskssequentially.Notably,both
theinstructionsandenvironmentsarenovel toourapproach.
Pleaserefer tothesupplementaryvideos.
material)underdiverseenvironments inazero-shotmanner.
Notably,boththeinstructionsandenvironmentsarenovel to
ourmethod.Wefirstevaluatedtheperformanceof individual
navigationtasks (Fig.5), including(A)vision-and-language
navigation, (B)objectgoalnavigation, (C)embodiedquestion
answering,and(D)humanfollowing.WefoundthatUni-NaVid
canunderstanddiverseinstructionsanddemonstratesimpressive
performanceinlong-horizonnavigationtasks(e.g.,navigating
acrosshallwaysandenteringrooms),aswellas insearching
forout-of-visionobjectsandansweringsubsequentquestions.
Moreover, theagent is capableof followingahumaneven
whentheperson’sappearancedeviates fromthedescription
of theavatar inthehuman-followingdataset (Sec.V-A).The
statisticsof thecorrespondingreal-worldexperimentscanbe
foundintheSupplementalMaterial.
Inadditiontoindividualnavigationtasks,wealsoevaluate
ourmethodonmorecomplexinstructions involvingmultiple
navigationtasks(Fig.6). Inthisscenario, theagent isrequired
tosequentiallyexecutethenavigationtasksdescribedinthe
language instructions.Ourmodel demonstrates impressiove
performance inaligningthecurrentnavigationprocesswith
theinstructions toreasonabout thecurrent stateofnavigation.
Furthermore, we provide a detailed illustration of action
predictionduringnavigation inFig. 7,wherewe plot the
predictedactionprobabilities ofUni-NaVid.Notably,with
onlyslightdifferences inobjectdescriptions, e.g., ’chairwith
a toy’ and’chairwithasweater’.Specifically, ourmethod
successfullydistinguishesbetweenthelocationsandpredicts
actionsaccordingly. Interestingly, theactionprobabilities(for
thenext fouractions) revealasequentialorderofactions:first
turningright/left, followedbymovingforward.Weprovide
additionalvisual resultsofourmethodintheSupplemental
Materialandencouragetheaudiencetoviewourvideo,which
showcases thereal-worldperformanceofourmethod.
D. AblationStudy
Visualizationof training strategy.Wepresent avisual
izationof thetrainingstrategy’sperformanceinFigure8. In
Fig. 8(a),wecompare trainingonasinglenavigation task
withtrainingacrossmultiple tasks.Theresultsdemonstrate
thesynergisticbenefitsofmulti-task learning,whichyields
consistent performance improvements across all navigation
tasks. Notably, VLN, ObjectNav, and EQAexhibit more
significant improvements,whileFollowingshows relatively
Move to a chairwith a toy, then 
turn right, move tothechair
withasweater.
Move to a chair with a sweater, 
then turn left, move tothe
chairwithatoy.
Third-person view Trajectory
Third-person view Trajectory
A
B
C
A
B
C
A
B
A
B
C C
Next Action 1 Next Action 2 Next Action 3 Next Action 4 Next Action 1 Next Action 2 Next Action 3 Next Action 4
1.0
0.5
0.0
1.0
0.5
0.0
A
B
C
A
B
C
Predicted Action Probability
Fig. 7:ActionpredictionontheVLNtasks.Weevaluate
Uni-NaVidonchallengingopen-vocabularyobjects,requiringit
torecognizethetargetobjectsandfollowthespecifiedmotions.
Weprovidethepredictedactionprobabilities(for thenext four
actions) todemonstrateitsbreak-innavigationcapability.
(b)Continual Improvement with Scale (a) Multi-task Synergy
Performance (%)
Navigation Tasks Training Samples
Fig.8:Comparsiononmulti-tasktraininganddatascale.(a)
Wepresent themulti-tasksynergyofourmethod, illustrating
theperformancecomparisonbetweentrainingwithasingle
taskandtrainingwithmultipletasks; (b)wedemonstratethe
performanceacrossdifferentnavigationtasksundervarying
numbersof trainingsamples.
smallergains.Weattributethisdifferencetothelowerreliance
of the Following task on historical context. Additionally,
we investigate the influence of data scale on navigation
performance (Figure 8 (b)).We observe that performance
improvesacrossallnavigationtaskswithlargerdatavolumes.
However, the incrementalgaindiminishes (from3Mto6M
samples), potentiallydue tolimitations inthedatadiversity
ofsimulators.Specifically, for theFollowingtask, thereason
for theslowerconvergenceis theheavyocclusioncausedby
obstaclesorotherhumans.Thishighlights theneedformore
high-quality followingdata samples,whichcanenableour
model tolearnmoreeffectivelyandperformbetter inhighly
dynamicenvironments.
Ablation on training strategy and architecture.We
conductexperimentstoevaluatetheeffectivenessofthetraining
Type VLN(SR↑)ObjNav(SR↑)EQA(ACC↑)Follow(SR↑)
No<Nav>token 35.2 69.1 20.4 55.1
NoVQAdata 40.5 50.6 1.19 58.8
Curr. 9.61 44.3 32.5 56.3
Curr.+Short. 39.7 67.8 44.1 59.7
Curr.+Short.+Long. 48.7 73.7 47.3 61.2
TABLEX:Ablationstudyontrainingstrategyandarchi
tecture.Foreachablationtype,weretraintheentiremodel
andevaluateitsperformanceacrossfournavigationtasks.
strategy and tokenmerging designs (Tab. X). Our results
indicate that theabsenceof<NAV>andVQAdata leads to
aperformancedeclineacrossall tasks, similarfindingscan
befoundin[14,111].Notably, theperformancedropismost
obviouslyinEQA, as thelackof<NAV>special tokenmakes
themodelmisinterpretwhether it shouldanswerquestionsor
output actions.Additionally,withoutVQAdata, theagent’s
abilitytoanswerquestionsdropssignificantly,almostrendering
it incapableofcorrectlyansweringquestions.Webelievethis
isduetothecatastrophicforgettingprobleminLLMs,where
themodel losesopen-worldknowledgebybeingtrainedsolely
onnavigation-relateddata.
Fromtheperformanceof differentmemorydesigns,we
findthatbothshort-termandlong-termmemoryvisual tokens
contributetoperformanceimprovements.Inparticular,theVLN
taskshows themost significant performancedrop(−80.3%
SR)whenvisualmemoryisremoved, as thelackofmemory
hinders thealignmentofvisualhistorywithinstructions.For
theFollowingtask, theabsenceofmemoryresults inonlya
minorperformancedecline(−8%SR),as this taskprimarily
reliesonrecent frames totrackthetarget.Additionalablation
studiesonarchitectureandhyperparametersareprovidedin
theSupplementaryMaterial.
VII. LIMITATIONS
Despitethepromisingresults,Uni-NaVidhasseveral limita
tions.First,Uni-NaVidis trainedandevaluatedonfourwell
definednavigationtasks,whilethereexistsalargebodyof lit
eratureoninsightfulandpracticalnavigationdatasets[86,112].
We believe that collecting data fromthese datasets could
further enhance the navigation capabilities of ourmethod.
Second,ourmethodisdesignedtoacquiremulti-tasknavigation
capabilitiesunder theassumptionthat therobot isofstandard
size(seeSectionV-A).Toextendit torobotsofgeneral sizes,
aconvincingapproachistoincorporatepriorknowledgeof the
robot’ssize,asdemonstratedin[24,29].Third,ourmethod
iscurrentlylimitedtopredictingsimpletrajectoriescomposed
ofashorthorizonof future low-leveldiscreteactions.This
limitationcouldbealleviatedbyextendingthemoel topredict
continuousandsmoothtrajectorieswithtechniquesfrommotion
planning[79,81]orautonomousdriving[18,52].
VIII. DISCUSSIONANDCONCLUSION
Inthispaper,weintroduceanefficientvision-language-action
(VLA)model,Uni-NaVid,designedtoacquiregeneralembod
iednavigationskills through learningmulti-tasknavigation
data. To efficiently encode the online-captured video sequences
during navigation, we develop an online visual token merging
mechanism that separately processes current observations, short
term observations, and long-term observations. This design
enables our approach to operate at an average speed of 5 Hz. We
also collect 3.6 million navigation data points across four highly
demanded embodied navigation tasks, including vision-and
language navigation, object goal navigation, embodied question
answering, and human following. Extensive experiments and
ablation studies demonstrate that our method achieves SOTA
level performance using only monocular videos as input,
highlighting our model’s superior capability in learning multiple
navigation tasks. Moreover, we deploy Uni-NaVid in real-world
environments, demonstrating impressive generalizability and
versatile navigation performance in real worlds.
Future works. Our work serves merely as a starting point of
general-purpose navigation, and we hope it will inspire future
directions in this field:
• Benchmarking. With the consistent development of em
bodied navigation, there is a growing need for general
purpose navigation benchmarking. Such a benchmark
would help researchers better position their work and
drive progress in the navigation community.
• Architecture. We would like to further enhance the prac
ticality of our architecture by tackling very long-horizon
tasks (e.g., navigating across buildings) and incorporating
advanced motion planning techniques [81, 18].
• Application. We would like to apply our method to applica
tions such as robotic guide dogs and home service robots.
Additionally, we are excited to extend this technique to
other embodied AI tasks, such as mobile manipulation
[110, 92, 55].
REFERENCES
[1] Dong An, Hanqing Wang, Wenguan Wang, Zun Wang,
Yan Huang, Keji He, and Liang Wang. Etpnav:
Evolving topological planning for vision-language nav
igation in continuous environments. arXiv preprint
arXiv:2304.03047, 2023.
[2] Peter Anderson, Angel Chang, Devendra Singh Chaplot,
Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun,
Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Mano
lis Savva, et al. On evaluation of embodied navigation
agents. arXiv preprint arXiv:1807.06757, 2018.
[3] Peter Anderson, Angel Chang, Devendra Singh Chaplot,
Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun,
Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Mano
lis Savva, et al. On evaluation of embodied navigation
agents. arXiv preprint arXiv:1807.06757, 2018.
[4] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce,
Mark Johnson, Niko S¨underhauf, Ian Reid, Stephen
Gould, and Anton Van Den Hengel. Vision-and-language
navigation: Interpreting visually-grounded navigation
instructions in real environments. In Proceedings of
the IEEE conference on computer vision and pattern
[5] Peter Anderson, Ayush Shrivastava, Joanne Truong,
Arjun Majumdar, Devi Parikh, Dhruv Batra, and Ste
fan Lee. Sim-to-real transfer for vision-and-language
navigation. In Conference on Robot Learning, pages
671–681. PMLR, 2021.
[6] RC Atkinson and RM Shiffrin. Human memory: A
proposed system and its control processes (vol. 2). The
Psychology of Learning and Motivation: Advances in
Research and Theory, pages 89–195, 1968.
[7] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and
Motoaki Kawanabe. Scanqa: 3d question answering
for spatial scene understanding. In proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pages 19129–19139, 2022.
[8] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao
Zhang, Christoph Feichtenhofer, and Judy Hoffman. To
ken merging: Your vit but faster. ArXiv, abs/2210.09461,
2022.
[9] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen
Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding,
Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2:
Vision-language-action models transfer web knowledge
to robotic control. arXiv preprint arXiv:2307.15818,
2023.
[10] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem,
and Juan Carlos Niebles. Activitynet: A large-scale
video benchmark for human activity understanding. In
Proceedings of the ieee conference on computer vision
and pattern recognition, pages 961–970, 2015.
[11] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej
Halber, Matthias Niebner, Manolis Savva, Shuran Song,
Andy Zeng, and Yinda Zhang. Matterport3d: Learning
from rgb-d data in indoor environments. In 2017
International Conference on 3D Vision (3DV), pages
667–676. IEEE, 2017.
[12] Devendra Singh Chaplot, Dhiraj Prakashchand Gandhi,
Abhinav Gupta, and Russ R Salakhutdinov. Object
goal navigation using goal-oriented semantic exploration.
Advances in Neural Information Processing Systems, 33:
4247–4258, 2020.
[13] David Chen and William B Dolan. Collecting highly
parallel data for paraphrase evaluation. In Proceedings
of the 49th annual meeting of the association for
computational linguistics: human language technologies,
pages 190–200, 2011.
[14] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun
Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi,
Vikas Chandra, Yunyang Xiong, and Mohamed Elho
seiny. Minigpt-v2: large language model as a unified
interface for vision-language multi-task learning. arXiv
preprint arXiv:2310.09478, 2023.
[15] Kevin Chen, Junshen K Chen, Jo Chuang, Marynel
V´azquez, and Silvio Savarese. Topological planning
with transformers for vision-and-language navigation. In
Proceedings of the IEEE/CVF Conference on Computer
recognition, pages 3674–3683, 2018.
Vision and Pattern Recognition, pages 11276–11286,
2021.
sion and language navigation. In Proceedings of the
[16] Peihao Chen, Dongyu Ji, Kunyang Lin, Runhao Zeng,
Thomas H Li, Mingkui Tan, and Chuang Gan. Weakly
supervised multi-granularity map learning for vision-and
language navigation. arXiv preprint arXiv:2210.07506,
2022.
[17] Peihao Chen, Xinyu Sun, Hongyan Zhi, Runhao Zeng,
Thomas H Li, Gaowen Liu, Mingkui Tan, and Chuang
Gan. Action-aware zero-shot robot navigation by
exploiting vision-and-language ability of foundation
models. arXiv preprint arXiv:2308.07997, 2023.
[18] Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao,
Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu,
and Xinggang Wang. Vadv2: End-to-end vectorized
autonomous driving via probabilistic planning. arXiv
preprint arXiv:2402.13243, 2024.
[19] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace,
Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon,
Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan
Yang, et al. Panda-70m: Captioning 70m videos with
multiple cross-modality teachers. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 13320–13331, 2024.
[20] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng,
Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan
Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al.
Vicuna: An open-source chatbot impressing gpt-4 with
90%* chatgpt quality. See https://vicuna. lmsys. org
(accessed 14 April 2023), 2023.
[21] Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan
Lee, Devi Parikh, and Dhruv Batra. Embodied question
answering. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 1–10,
2018.
[22] Vishnu Sashank Dorbala, Gunnar Sigurdsson, Robinson
Piramuthu, Jesse Thomason, and Gaurav S Sukhatme.
Clip-nav: Using clip for zero-shot vision-and-language
navigation. arXiv preprint arXiv:2211.16649, 2022.
[23] Jiafei Duan, Samson Yu, Hui Li Tan, Hongyuan Zhu, and
Cheston Tan. A survey of embodied ai: From simulators
to research tasks. IEEE Transactions on Emerging Topics
in Computational Intelligence, 6(2):230–244, 2022.
[24] Ainaz Eftekhar, Luca Weihs, Rose Hendrix, Ege Caglar,
Jordi Salvador, Alvaro Herrasti, Winson Han, Eli Van
derBil, Aniruddha Kembhavi, Ali Farhadi, et al. The
one ring: a robotic indoor navigation generalist. arXiv
preprint arXiv:2412.14401, 2024.
[25] Anthony Francis, Claudia P´erez-d’Arpino, Chengshu Li,
Fei Xia, Alexandre Alahi, Rachid Alami, Aniket Bera,
Abhijat Biswas, Joydeep Biswas, Rohan Chandra, et al.
Principles and guidelines for evaluating social robot
navigation algorithms. arXiv preprint arXiv:2306.16740,
2023.
[26] Georgios Georgakis, Karl Schmeckpeper, Karan Wan
choo, Soham Dan, Eleni Miltsakaki, Dan Roth, and
Kostas Daniilidis. Cross-modal map learning for vi
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 15460–15470, 2022.
[27] Theophile Gervet, Soumith Chintala, Dhruv Batra, Ji
tendra Malik, and Devendra Singh Chaplot. Navigating
to objects in the real world. Science Robotics, 8(79):
eadf6991, 2023.
[28] Meenakshi Gupta, Swagat Kumar, Laxmidhar Behera,
and Venkatesh K Subramanian. A novel vision-based
tracking algorithm for a human-following mobile robot.
IEEE Transactions on Systems, Man, and Cybernetics:
Systems, 47(7):1415–1427, 2016.
[29] Noriaki Hirose, Dhruv Shah, Ajay Sridhar, and Sergey
Levine. Exaug: Robot-conditioned navigation policies
via geometric experience augmentation. In 2023 IEEE
International Conference on Robotics and Automation
(ICRA), pages 4077–4084. IEEE, 2023.
[30] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu,
Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang,
Junhui Ji, Zhao Xue, Lei Zhao, Zhuoyi Yang, Xiaotao
Gu, Xiaohan Zhang, Guanyu Feng, Da Yin, Zihan
Wang, Ji Qi, Xixuan Song, Peng Zhang, De-Feng
Liu, Bin Xu, Juanzi Li, Yu-Chen Dong, and Jie Tang.
Cogvlm2: Visual language models for image and video
understanding. ArXiv, abs/2408.16500, 2024. URL
https://api.semanticscholar.org/CorpusID:272146264.
[31] Yicong Hong, Zun Wang, Qi Wu, and Stephen Gould.
Bridging the gap between learning in discrete and
continuous environments for vision-and-language nav
igation. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages
15439–15449, 2022.
[32] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong
Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d
llm: Injecting the 3d world into large language models.
Advances in Neural Information Processing Systems, 36:
20482–20494, 2023.
[33] Shanee S Honig, Tal Oron-Gilad, Hanan Zaichyk, Vardit
Sarne-Fleischmann, Samuel Olatunji, and Yael Edan.
Toward socially aware person-following robots. IEEE
Transactions on Cognitive and Developmental Systems,
10(4):936–954, 2018.
[34] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram
Burgard. Visual language maps for robot navigation.
arXiv preprint arXiv:2210.05714, 2022.
[35] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram
Burgard. Visual language maps for robot navigation. In
2023 IEEE International Conference on Robotics and
Automation (ICRA), pages 10608–10615. IEEE, 2023.
[36] Yulong Huang, Yonggang Zhang, Peng Shi, Zhemin Wu,
Junhui Qian, and Jonathon A Chambers. Robust kalman
f
ilters based on gaussian scale mixture distributions with
application to target tracking. IEEE Transactions on
Systems, Man, and Cybernetics: Systems, 49(10):2082
2096, 2017.
[37] Md Jahidul Islam, Jungseok Hong, and Junaed Sattar.
Person-following by autonomous robots: A categorical
vision-and-language navigation with dense spatiotempo
overview. The International Journal of Robotics Re
search, 38(14):1581–1618, 2019.
[38] Md Mofijul Islam, Alexi Gladstone, Riashat Islam, and
Tariq Iqbal. Eqa-mx: Embodied question answering us
ing multimodal expression. In The Twelfth International
Conference on Learning Representations, 2023.
[39] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun
Cao, and Li Yuan. Chat-univi: Unified visual represen
tation empowers large language models with image and
video understanding. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 13700–13710, 2024.
[40] Abhishek Kadian, Joanne Truong, Aaron Gokaslan,
Alexander Clegg, Erik Wijmans, Stefan Lee, Manolis
Savva, Sonia Chernova, and Dhruv Batra. Sim2real
predictivity: Does evaluation in simulation predict real
world performance? IEEE Robotics and Automation
Letters, 5(4):6670–6677, 2020.
[41] Linh K¨astner, Bassel Fatloun, Zhengcheng Shen, Daniel
Gawrisch, and Jens Lambrecht. Human-following
and-guiding in crowded environments using semantic
deep-reinforcement-learning for mobile service robots.
In 2022 International Conference on Robotics and
Automation (ICRA), pages 833–839, 2022.
[42] Mukul Khanna, Yongsen Mao, Hanxiao Jiang, Sanjay
Haresh, Brennan Shacklett, Dhruv Batra, Alexander
Clegg, Eric Undersander, Angel X Chang, and Manolis
Savva. Habitat synthetic scenes dataset (hssd-200): An
analysis of 3d scene scale and realism tradeoffs for
objectgoal navigation. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 16384–16393, 2024.
[43] Jacob Krantz and Stefan Lee. Sim-2-sim transfer for
vision-and-language navigation in continuous environ
ments. In European Conference on Computer Vision,
pages 588–603. Springer, 2022.
[44] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv
Batra, and Stefan Lee. Beyond the nav-graph: Vision
and-language navigation in continuous environments. In
European Conference on Computer Vision, 2020. URL
https://api.semanticscholar.org/CorpusID:214802389.
[45] Jacob Krantz, Aaron Gokaslan, Dhruv Batra, Stefan
Lee, and Oleksandr Maksymets. Waypoint models
for instruction-guided navigation in continuous environ
ments. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 15162–15171,
2021.
[46] Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie,
and Jason Baldridge. Room-across-room: Multilingual
vision-and-language navigation with dense spatiotempo
ral grounding. In Proceedings of the 2020 Conference
on Empirical Methods in Natural Language Processing
(EMNLP), pages 4392–4412, 2020.
[47] Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie,
and Jason Baldridge. Room-across-room: Multilingual
ral grounding. In Proceedings of the 2020 Conference
on Empirical Methods in Natural Language Processing
(EMNLP), pages 4392–4412, 2020.
[48] Yuxuan Kuang, Hai Lin, and Meng Jiang. Openfm
nav: Towards open-set zero-shot object navigation via
vision-language foundation models. arXiv preprint
arXiv:2402.10670, 2024.
[49] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai
Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao.
Videochat: Chat-centric video understanding. arXiv
preprint arXiv:2305.06355, 2023.
[50] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang,
Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al.
Mvbench: A comprehensive multi-modal video under
standing benchmark. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 22195–22206, 2024.
[51] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid:
An image is worth 2 tokens in large language models.
arXiv preprint arXiv:2311.17043, 2023.
[52] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang,
Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li,
Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated
diffusion model for end-to-end autonomous driving.
arXiv preprint arXiv:2411.15139, 2024.
[53] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng
Jin, and Li Yuan. Video-llava: Learning united visual
representation by alignment before projection. arXiv
preprint arXiv:2311.10122, 2023.
[54] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. Visual instruction tuning. In NeurIPS, 2023.
[55] Peiqi Liu, Yaswanth Orru, Chris Paxton, Nur Muham
mad Mahi Shafiullah, and Lerrel Pinto. Ok-robot: What
really matters in integrating open-knowledge models for
robotics. arXiv preprint arXiv:2401.12202, 2024.
[56] Ruyang Liu, Chen Li, Yixiao Ge, Thomas H Li, Ying
Shan, and Ge Li. Bt-adapter: Video conversation is
feasible without video instruction tuning. In Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 13658–13667, 2024.
[57] Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge, Ying
Shan, and Ge Li. St-llm: Large language models are
effective temporal learners. In European Conference on
Computer Vision, pages 1–18. Springer, 2025.
[58] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao
Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei
Yang, Hang Su, et al. Grounding dino: Marrying dino
with grounded pre-training for open-set object detection.
arXiv preprint arXiv:2303.05499, 2023.
[59] Yang Liu, Weixing Chen, Yongjie Bai, Xiaodan Liang,
Guanbin Li, Wen Gao, and Liang Lin. Aligning cyber
space with physical world: A comprehensive survey on
embodied ai. arXiv preprint arXiv:2407.06886, 2024.
[60] Yuxing Long, Xiaoqi Li, Wenzhe Cai, and Hao
Dong. Discuss before moving: Visual language nav
igation via multi-expert discussions. arXiv preprint
search strategies from human demonstrations at scale. In
arXiv:2309.11382, 2023.
[61] Yuxing Long, Wenzhe Cai, Hongcheng Wang, Guanqi
Zhan, and Hao Dong. Instructnav: Zero-shot system for
generic instruction navigation in unexplored environment.
arXiv preprint arXiv:2406.04882, 2024.
[62] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and
Fahad Shahbaz Khan. Video-chatgpt: Towards detailed
video understanding via large vision and language
models. arXiv preprint arXiv:2306.05424, 2023.
[63] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav
Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal,
Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al.
Openeqa: Embodied question answering in the era of
foundation models. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 16488–16498, 2024.
[64] Christoforos Mavrogiannis, Francesca Baldini, Allan
Wang, Dapeng Zhao, Pete Trautman, Aaron Steinfeld,
and Jean Oh. Core challenges of social robot naviga
tion: A survey. ACM Transactions on Human-Robot
Interaction, 12(3):1–39, 2023.
[65] Wentao Mo and Yang Liu. Bridging the gap between
2d and 3d visual question answering: A fusion approach
for 3d vqa. In Proceedings of the AAAI Conference
on Artificial Intelligence, volume 38, pages 4261–4268,
2024.
[66] Khanh Nguyen, Debadeepta Dey, Chris Brockett, and
Bill Dolan. Vision-based navigation with language-based
assistance via imitation learning with indirect interven
tion. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 12527
12537, 2019.
[67] OpenAI. Gpt-4 technical report, 2023.
[68] Xavier Puig, Eric Undersander, Andrew Szot,
Mikael Dallaire Cote, Tsung-Yen Yang, Ruslan Partsey,
Ruta Desai, Alexander William Clegg, Michal Hlavac,
So Yeon Min, et al. Habitat 3.0: A co-habitat for humans,
avatars and robots. arXiv preprint arXiv:2310.13724,
2023.
[69] Santhosh K Ramakrishnan, Aaron Gokaslan, Erik Wij
mans, Oleksandr Maksymets, Alex Clegg, John Turner,
Eric Undersander, Wojciech Galuba, Andrew Westbury,
Angel X Chang, et al. Habitat-matterport 3d dataset
(hm3d): 1000 large-scale 3d environments for embodied
ai. arXiv preprint arXiv:2109.08238, 2021.
[70] Santhosh Kumar Ramakrishnan, Aaron Gokaslan, Erik
Wijmans, Oleksandr Maksymets, Alexander Clegg,
John M Turner, Eric Undersander, Wojciech Galuba,
Andrew Westbury, Angel X Chang, et al. Habitat
matterport 3d dataset (hm3d): 1000 large-scale 3d
environments for embodied ai. In Thirty-fifth Conference
on Neural Information Processing Systems Datasets and
Benchmarks Track (Round 2), 2021.
[71] Ram Ramrakhya, Eric Undersander, Dhruv Batra, and
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 5173–5183, 2022.
[72] Ram Ramrakhya, Dhruv Batra, Erik Wijmans, and
Abhishek Das. Pirlnav: Pretraining with imitation and
rl finetuning for objectnav. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 17896–17906, 2023.
[73] Sonia Raychaudhuri, Saim Wani, Shivansh Patel, Un
nat Jain, and Angel X Chang. Language-aligned
waypoint (law) supervision for vision-and-language
navigation in continuous environments. arXiv preprint
arXiv:2109.15207, 2021.
[74] St´ephane Ross, Geoffrey Gordon, and Drew Bagnell. A
reduction of imitation learning and structured prediction
to no-regret online learning. In Proceedings of the four
teenth international conference on artificial intelligence
and statistics, pages 627–635. JMLR Workshop and
Conference Proceedings, 2011.
[75] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets,
Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub,
Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh,
and Dhruv Batra. Habitat: A Platform for Embodied AI
Research. ICCV, 2019.
[76] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets,
Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub,
Jia Liu, Vladlen Koltun, Jitendra Malik, et al. Habitat:
A platform for embodied ai research. In Proceedings of
the IEEE/CVF International Conference on Computer
Vision, pages 9339–9347, 2019.
[77] James A Sethian. Fast marching methods. SIAM review,
41(2):199–235, 1999.
[78] Dhruv Shah, Bła˙zej Osi´nski, Sergey Levine, et al. Lm
nav: Robotic navigation with large pre-trained models
of language, vision, and action. In Conference on robot
learning, pages 492–504. PMLR, 2023.
[79] Dhruv Shah, Ajay Sridhar, Arjun Bhorkar, Noriaki
Hirose, and Sergey Levine. Gnm: A general navigation
model to drive any robot. In 2023 IEEE International
Conference on Robotics and Automation (ICRA), pages
7226–7233. IEEE, 2023.
[80] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng
Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun
Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From
dense token to sparse memory for long video under
standing. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages
18221–18232, 2024.
[81] Ajay Sridhar, Dhruv Shah, Catherine Glossop, and
Sergey Levine. Nomad: Goal masked diffusion policies
for navigation and exploration. In 2024 IEEE Interna
tional Conference on Robotics and Automation (ICRA),
pages 63–70. IEEE, 2024.
[82] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and
Yue Cao. Eva-clip: Improved training techniques for
Abhishek Das. Habitat-web: Learning embodied object
clip at scale. arXiv preprint arXiv:2303.15389, 2023.
[83] Sinan Tan, Mengmeng Ge, Di Guo, Huaping Liu, and
information: A survey from tasks to methodology.
Fuchun Sun. Knowledge-based embodied question
answering. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 45(10):11948–11960, 2023.
[84] Nguyen Van Toan, Minh Do Hoang, Phan Bui Khoi,
and Soo-Yeong Yi. The human-following strategy for
mobile robots in mixed environments. Robotics and
Autonomous Systems, 160:104317, 2023.
[85] Hanqing Wang, Wei Liang, Luc V Gool, and Wenguan
Wang. Towards versatile embodied navigation. Advances
in neural information processing systems, 35:36858
36874, 2022.
[86] Hongcheng Wang, Andy Guan Hong Chen, Xiaoqi
Li, Mingdong Wu, and Hao Dong. Find what you
want: learning demand-conditioned object attribute space
for demand-driven navigation. Advances in Neural
Information Processing Systems, 36, 2024.
[87] Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu,
and Shuqiang Jiang. Gridmm: Grid memory map for
vision-and-language navigation. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 15625–15636, 2023.
[88] Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu, and
Shuqiang Jiang. Sim-to-real transfer via 3d feature
f
ields for vision-and-language navigation. arXiv preprint
arXiv:2406.09798, 2024.
[89] Zun Wang, Jialu Li, Yicong Hong, Yi Wang, Qi Wu,
Mohit Bansal, Stephen Gould, Hao Tan, and Yu Qiao.
Scaling data generation in vision-and-language naviga
tion. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 12009–12020,
2023.
[90] Erik Wijmans, Samyak Datta, Oleksandr Maksymets,
Abhishek Das, Georgia Gkioxari, Stefan Lee, Irfan Essa,
Devi Parikh, and Dhruv Batra. Embodied question
answering in photorealistic environments with point
cloud perception. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 6659–6668, 2019.
[91] Erik Wijmans, Abhishek Kadian, Ari Morcos, Stefan Lee,
Irfan Essa, Devi Parikh, Manolis Savva, and Dhruv Batra.
DD-PPO: Learning near-perfect pointgoal navigators
from 2.5 billion frames. In International Conference on
Learning Representations (ICLR), 2020.
[92] Jimmy Wu, William Chong, Robert Holmberg, Aaditya
Prasad, Yihuai Gao, Oussama Khatib, Shuran Song,
Szymon Rusinkiewicz, and Jeannette Bohg. Tidybot++:
An open-source holonomic mobile manipulator for robot
learning. arXiv preprint arXiv:2412.10447, 2024.
[93] Qiaoyun Wu, Xiaoxi Gong, Kai Xu, Dinesh Manocha,
Jingxuan Dong, and Jun Wang. Towards target-driven
visual navigation in indoor scenes via generative imita
tion learning. IEEE Robotics and Automation Letters, 6
(1):175–182, 2020.
[94] Yuchen Wu, Pengcheng Zhang, Meiying Gu, Jin Zheng,
and Xiao Bai. Embodied navigation with multi-modal
Information Fusion, page 102532, 2024.
[95] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang
Zhang, Xiangnan He, and Yueting Zhuang. Video
question answering via gradually refined attention over
appearance and motion. In ACM Multimedia, 2017.
[96] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A
large video description dataset for bridging video and
language. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 5288
5296, 2016.
[97] Zifan Xu, Bo Liu, Xuesu Xiao, Anirudh Nair, and Peter
Stone. Benchmarking reinforcement learning techniques
for autonomous navigation. In 2023 IEEE International
Conference on Robotics and Automation (ICRA), pages
9224–9230. IEEE, 2023.
[98] Karmesh Yadav, Arjun Majumdar, Ram Ramrakhya,
Naoki Yokoyama, Alexei Baevski, Zsolt Kira, Oleksandr
Maksymets, and Dhruv Batra. Ovrl-v2: A simple state-of
art baseline for imagenav and objectnav. arXiv preprint
arXiv:2303.07798, 2023.
[99] Karmesh Yadav, Ram Ramrakhya, Arjun Majumdar,
Vincent-Pierre Berges, Sachit Kuhar, Dhruv Batra,
Alexei Baevski, and Oleksandr Maksymets. Offline
visual representation learning for embodied navigation.
In Workshop on Reincarnating Reinforcement Learning
at ICLR 2023, 2023.
[100] Naoki Yokoyama, Sehoon Ha, Dhruv Batra, Jiuguang
Wang, and Bernadette Bucher. Vlfm: Vision-language
frontier maps for zero-shot semantic navigation. In
2024 IEEE International Conference on Robotics and
Automation (ICRA), pages 42–48. IEEE, 2024.
[101] Naoki Yokoyama, Ram Ramrakhya, Abhishek Das,
Dhruv Batra, and Sehoon Ha. Hm3d-ovon: A dataset and
benchmark for open-vocabulary object goal navigation.
arXiv preprint arXiv:2409.14296, 2024.
[102] Bangguo Yu, Hamidreza Kasaei, and Ming Cao. L3mvn:
Leveraging large language models for visual target
navigation. In 2023 IEEE/RSJ International Conference
on Intelligent Robots and Systems (IROS), pages 3554
3560. IEEE, 2023.
[103] Licheng Yu, Xinlei Chen, Georgia Gkioxari, Mohit
Bansal, Tamara L Berg, and Dhruv Batra. Multi-target
embodied question answering. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 6309–6318, 2019.
[104] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao,
Yueting Zhuang, and Dacheng Tao. Activitynet-qa:
A dataset for understanding complex web videos via
question answering. In Proceedings of the AAAI
Conference on Artificial Intelligence, volume 33, pages
9127–9134, 2019.
[105] Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, and Qi Tian.
Deep modular co-attention networks for visual question
answering. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 6281
6290, 2019.
and Yizhou Wang. Towards distraction-robust active
[106] Kuo-Hao Zeng, Zichen Zhang, Kiana Ehsani, Rose
Hendrix, Jordi Salvador, Alvaro Herrasti, Ross Girshick,
Aniruddha Kembhavi, and Luca Weihs. Poliformer: Scal
ing on-policy rl with transformers results in masterful
navigators. arXiv preprint arXiv:2406.20083, 2024.
[107] Hang Zhang, Xin Li, and Lidong Bing. Video-llama:
An instruction-tuned audio-visual language model for
video understanding. arXiv preprint arXiv:2306.02858,
2023.
[108] Jiazhao Zhang, Chenyang Zhu, Lintao Zheng, and
Kai Xu. Rosefusion: random optimization for online
dense reconstruction under fast camera motion. ACM
Transactions on Graphics (TOG), 40(4):1–17, 2021.
[109] Jiazhao Zhang, Yijie Tang, He Wang, and Kai Xu. Asro
dio: Active subspace random optimization based depth
inertial odometry. IEEE Transactions on Robotics, 39
(2):1496–1508, 2022.
[110] Jiazhao Zhang, Nandiraju Gireesh, Jilong Wang, Xi
aomeng Fang, Chaoyi Xu, Weiguang Chen, Liu Dai,
and He Wang. Gamma: Graspability-aware mobile
manipulation policy learning based on online grasping
pose fusion. In 2024 IEEE International Conference
on Robotics and Automation (ICRA), pages 1399–1405.
IEEE, 2024.
[111] Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze
Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng
Zhang, and He Wang. Navid: Video-based vlm plans the
next step for vision-and-language navigation. Robotics:
Science and Systems, 2024.
[112] Yue Zhang, Ziqiao Ma, Jialu Li, Yanyuan Qiao, Zun
Wang, Joyce Chai, Qi Wu, Mohit Bansal, and Parisa
Kordjamshidi. Vision-and-language navigation today
and tomorrow: A survey in the era of foundation
models. ArXiv, abs/2407.07035, 2024. URL https:
//api.semanticscholar.org/CorpusID:271064503.
[113] Zhen Zhang, Jiaqing Yan, Xin Kong, Guangyao Zhai,
and Yong Liu. Efficient motion planning based on
kinodynamic model for quadruped robots following
persons in confined spaces. IEEE/ASME Transactions
on Mechatronics, 26(4):1997–2006, 2021.
[114] Duo Zheng, Shijia Huang, Lin Zhao, Yiwu Zhong, and
Liwei Wang. Towards learning a generalist model for
embodied navigation. arXiv preprint arXiv:2312.02010,
2023.
[115] Lintao Zheng, Chenyang Zhu, Jiazhao Zhang, Hang
Zhao, Hui Huang, Matthias Niessner, and Kai Xu. Active
scene understanding via online semantic reconstruction.
In Computer Graphics Forum, volume 38, pages 103
114. Wiley Online Library, 2019.
[116] Fangwei Zhong, Peng Sun, Wenhan Luo, Tingyun Yan,
and Yizhou Wang. Ad-vat+: An asymmetric dueling
mechanism for learning and understanding visual active
tracking. IEEE transactions on pattern analysis and
machine intelligence, 43(5):1467–1482, 2019.
visual tracking. In International Conference on Machine
Learning, pages 12782–12792. PMLR, 2021.
[118] Fangwei Zhong, Xiao Bi, Yudi Zhang, Wei Zhang,
and Yizhou Wang. Rspt: reconstruct surroundings and
predict trajectory for generalizable active object tracking.
In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 37, pages 3705–3714, 2023.
[119] Fangwei Zhong, Kui Wu, Hai Ci, Churan Wang, and
Hao Chen. Empowering embodied visual tracking
with visual foundation models and offline rl. In
European Conference on Computer Vision, pages 139
155. Springer, 2024.
[120] Gengze Zhou, Yicong Hong, and Qi Wu. Navgpt: Ex
plicit reasoning in vision-and-language navigation with
large language models. arXiv preprint arXiv:2305.16986,
2023.
[121] Gengze Zhou, Yicong Hong, Zun Wang, Xin Eric
Wang, and Qi Wu. Navgpt-2: Unleashing navigational
reasoning capability for large vision-language models.
In European Conference on Computer Vision, pages
260–278. Springer, 2025.
[122] Deyao Zhu, Jun Chen, Kilichbek Haydarov, Xiaoqian
Shen, Wenxuan Zhang, and Mohamed Elhoseiny. Chat
gpt asks, blip-2 answers: Automatic questioning towards
enriched visual descriptions, 2023.
[123] Fengda Zhu, Yi Zhu, Vincent Lee, Xiaodan Liang, and
Xiaojun Chang. Deep learning for embodied vision
navigation: A survey. arXiv preprint arXiv:2108.04097,
2021.
[124] Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J Lim,
Abhinav Gupta, Li Fei-Fei, and Ali Farhadi. Target
driven visual navigation in indoor scenes using deep
reinforcement learning. In 2017 IEEE international
conference on robotics and automation (ICRA), pages
3357–3364. IEEE, 2017.
[125] Weiqin Zu, Wenbin Song, Ruiqing Chen, Ze Guo,
Fanglei Sun, Zheng Tian, Wei Pan, and Jun Wang.
Language and sketching: An llm-driven interactive
multimodal multitask robot navigation framework. In
2024 IEEE International Conference on Robotics and
Automation (ICRA), pages 1019–1025. IEEE, 2024.
[117] Fangwei Zhong, Peng Sun, Wenhan Luo, Tingyun Yan,
CONTENTS
I Introduction 1
II RelatedWorks 2
III ProblemFormulation 3
IV ModelofUni-NaVid 4
IV-A ObservationEncoding. . . . . . . . . . 4
IV-B OnlineVisualTokenMerging. . . . . . 4
IV-C ActionPlanning . . . . . . . . . . . . . 5
V DataCollectionandTraining 5
V-A Multi-TaskNavigationData. . . . . . . 6
V-B TrainingStrategyofUni-NaVid . . . . 7
VI Experiment 7
VI-A DeploymentDetailsofUni-Navid. . . . 7
VI-B IndividualTaskResults . . . . . . . . . 8
VI-C QualitativeResults inReal-World . . . 9
VI-D AblationStudy. . . . . . . . . . . . . . 10
VII Limitations 11
VIII DiscussionandConclusion 11
IX TaskDefinition 18
IX-A Vision-and-languageNavigation(VLN) 18
IX-B ObjectGoalNavigation(ObjectNav) . . 18
IX-C EmbodiedQuestionAnswering(EQA) . 18
IX-D HumanFollowing . . . . . . . . . . . . 19
X ImplementationDetails 19
X-A OnlineVisualTokenMerging. . . . . . 19
X-B TokenOrganization . . . . . . . . . . . 19
X-C TraingStrategy . . . . . . . . . . . . . 19
XI DataPreparationDetails 19
XI-A DataCollection . . . . . . . . . . . . . 19
XI-B InstructionAugmentation . . . . . . . . 20
XII Real-worlddeployment 20
XII-A RobotSetup. . . . . . . . . . . . . . . . 20
XII-B Real-worldSystemArchitecture . . . . 21
XIII Experiments 21
XIII-A ExperimentSetup . . . . . . . . . . . . 21
XIII-A1 Benchmark. . . . . . . . . . 21
XIII-A2 Metrics. . . . . . . . . . . . 21
XIII-B Real-worldExperiments . . . . . . . . . 22
XIII-C MoreBenchmarkExperiments . . . . . 22
XIII-D MoreAblationstudy . . . . . . . . . . 23
XIII-E TimeAnalyze . . . . . . . . . . . . . . 23
XIII-F QualitativeExperiments . . . . . . . . . 23
IX. TASKDEFINITION
Weintroducethedetailsof fourembodiednavigationtasks
thatareincludedinourpaper.
A. Vision-and-languageNavigation(VLN)
Vision-and-languagenavigation[44] requires theagent to
followtheinstructionbymovingbetweengivenlandmarksand
stoppingat thedescribeddestinationwithinunseenenviron
ments.TheinstructionofVLNisat free-formwhichdescribes
a trajectory of landmarks and themotions between these
landmarks.Thelandmarksandmotionsareopen-vocabulary.
VLNiswidelyregardedasachallengingtaskbecauseithas
tounderstandthefree-forminstruction, alignthenavigation
historywith instruction, anddopathplanning.Despite the
fact that manyworks consider using pre-build landmark
graphs[120,61,121] tosimplifyVLN,weconsideramore
practical setting that uses acontinuous environment (VLN
CE [44]). Currently, there are twomainstreamVLN-CE
datasets:VLN-CER2R[44] andVLN-CERxR[47].We
provideexamplesof theseinstructions:
•VLN-CER2R:Walkdownthehallwayalongthebanister
railingontheupperfloorof thehome.Walkthroughthe
opendoornext tothestaircase.Walkintotheroom,which
hasacouchandchairsaroundacoffeetable.
•VLN-CERxR:Youarestartinginthecornerofaliving
room.Turnaroundtofindaclockhangingonthewall
inthehallway.Taketwosteps towardit.Turnrightand
walkstraight, passingbetween thebluecouchand the
kitchen.Youshouldnowbe looking throughwindows
intothebackyard.Toyour right isanopenpatio, andto
yourleftarefourframedblack-and-whitepictures.You’ve
completedtheinstructions.
In thebenchmarkVLN-CER2RandVLN-CERxR, the
agent is required tonavigateeachof the landmarks in the
givenorderof theinstructionandstopwithin3metersof the
destination.Themaxnavigationstepis500steps.
B. ObjectGoalNavigation(ObjectNav)
InObjectNav[76],anagentisinitializedatarandomstarting
positionandorientationinanunseenenvironmentandasked
tofindaninstanceofanobject category(‘findachair’)by
navigatingtoit.Theagent shouldexploretheenvironments to
trackthelocationlocationthatcouldlocatetargetobjects, and
thenidentifythetargetobjectandstopnearby.Specifically,we
followtheHabitat-matterport3Ddataset,whichrequires the
agent tosearchaspecificcategoryobject,whichcouldbeone
ofacategoryset includingcouch,bed, chair, toilet,plant, and
TV. InthebenchmarkHM3D, thetrajectoryisconsidereda
success if thetarget stopswithin1meterof thetargetobject
under500steps.
C. EmbodiedQuestionAnswering(EQA)
EmbodiedQuestionAnswering[21] isacomplicatedtask,
whichinvolvesansweringquestionsbyinteractingwithand
navigatingwithinanunseen3Denvironment.Followingthe
settingofMP3D-EQA,theagentisfirstgivenaquestionrelated
toanobject, likethecoloror location,andthentheagent is
requiredtoidentifythedescribedtargetandreturnanatural
language that directlyanswers thequestion.Herearesome
examplesof thequestions:
frames. When the model only receives one frame, that frame
Grid 
Pooling
ViT
Observed RGB image
…
…
8
Current Visual tokens
Grid 
Pooling
Grid 
Pooling
Shor-term 
visual tokens
Long-term 
visual tokens
Fig. 9: Grid pooling. We add a visualization of grid pooling
across different types of observations.
• What room is the chair located in?
• What color is the bed?
• What color is the couch in the living room?
In the benchmark MP3D-EQA, the agent is required to find
the target and answer the question within 500 steps. And the
episode will be considered a success if the answer matches the
GT answer.
D. Human Following
The three tasks discussed above primarily address static
and fixed scenarios, with limited applicability to dynamic
environments. In contrast, human following [28, 37, 125], as
a classic robot target tracking problem, focuses on enabling
robots to interact effectively in dynamic settings, making it
particularly well-suited for real-world applications.
Traditional human following tasks often rely on modular
based approaches [33], mainly focusing on the robot’s following
strategies [84] and controller design [41]. These studies
typically assume that the target to be tracked is either known or
specified at the beginning, an assumption that fails to account
for the complexities of real-world environments. In this paper,
we introduce a novel human-following task driven by natural
language instruction, where the robot should locate and follow
a specific human that aligns with the given description in a
potentially crowded and dynamic environment.
We developed a benchmark for this task based on Habitat
3.0 [68]. In each episode, multiple humans with diverse
characteristics are randomly placed in the scene, and the robot
is initialized near the target human to be followed, ensuring that
this human is within the robot’s initial observation. The robot
interprets natural language instructions, such as “follow the
man wearing a blue shirt and black pants” or “stay behind the
woman in yellow”, to locate the individual that best matches the
description. It then tracks and follows the person’s movements,
dynamically adapting until reaching the pre-defined navigation
goal. An episode is deemed successful if the robot stops within
2 meters of the correct human and faces him/her at the end.
X. IMPLEMENTATION DETAILS
In this section, we provide more implementation details of
Uni-NaVid.
A. Online Visual Token Merging
Online visual token merging is the key technique of Uni
NaVid to enable efficient encoding of a long horizon of ego
…
becomes the current observation. Then we extract the visual
tokens and leverage grid pooling (See Fig. 9). We split the
image into 8×8 and ach grid is conducted an average operation,
leading to the final 64 tokens for current observations. Then
with more incoming frames, we perform grid pooling to older
current observation tokens, which leads to 2 × 2 tokens, and
append them into a short-term memory buffer.
If the time step is over 65 steps, then the oldest short-term
frame will be pooped out and then is performed grid pooling to
1 token. This token is a long-term visual token which then will
be inserted into the long-term visual token list if the long-term
visual token list is empty or the cos similarity is smaller than
τ. Here, we use τ = 0.95, which is obtained empirically [80],
which achieves a balance between the efficiency and effective.
B. Token Organization
To facilitate the understanding of the large language model,
we have to organize the tokens, including observation tokens,
instruction tokens, and special tokens. Specifically, we use
observation indicator tokens to indicate the following parts
are visual tokens. Besides, we add an image separator token
between the adjacent visual tokens of each frame (following
existing VLMs [30, 57]), this is crucial to distinguish the visual
information inherent from different frames. Finally, if the task
is navigation, we add a navigation special token <NAV> to
indicate the task is navigation. Once the model understands the
navigation special token, it will directly output action tokens.
Note that, it is important to use a navigation special token
which could address the ambiguity problem under the embodied
question-answering task because a large language model could
confused about whether to directly answer the question or
output actions to the target. The supported experiments can be
found in the main paper Table 9.
C. Traing Strategy
Wefollow the training strategy of NaVid [111] in a two-stage
manner. In the first stage, we firstly pre-train the projector of
Uni-NaVid with the Image QA dataset then finetune both the
projector and LLM with the Video QA dataset. We collect the
data from LLama-Vid [51] and Pandm [19]. In the second state,
we train both projector and LLM with collected navigation
data. The default parameters are borrowed from NaVid [111].
XI. DATA PREPARATION DETAILS
To train Uni-NaVid, we required massive navigation data
across different navigation tasks. It is extremely challenging to
collect a large number of high-quality annotated data in the real
world. Therefore, we collect navigation data in the synthetic
environments and we describe the details in the following
sections.
A. Data Collection
Navigation samples. We define the navigation samples
including a navigation history video, corresponding instructions,
centric videos. Specifically, we organize the online captured
and future actions (we use four actions). Here the navigation
Portable Wi-Fi
history video is accumulated frames to time step T, which can
be indicated as a set of frames {xt}1:T
Vision-and-language navigation. We collect VLN navigation
samples in the VLN-CE simulator [44, 75]. We use the
training split of R2R dataset [44] and RxR dataset [46],
which include the ground-truth navigation trajectory and
corresponding instructions within the HM3D datasets [70].
Therefore, we deploy the agent to follow the GT trajectory
and render RGB images during navigation. In this case, we
collect 0.64 M navigation video-action samples.
Besides GT navigation samples, we also use DAGGER [74]
to collect more diverse navigation samples, nearly 1.69 M
navigation samples. Specifically, we run Uni-NaVidin the
training split and collect the expert actions by using a
deterministic path planning method [77] to the next non-arrived
landmark.
Wealso collect a sub-split (70k) of previously collected VLN
data (randomly sample the trajectories which are smaller than
20 steps) and augment the instruction with low-level actions,
e.g., “move forward 4 steps, then turn right 3 steps.”. We find
that Uni-NaVidcan easily master the low-level instructions, but
the performance drops when the low-level instruction expands
significantly.
Object goal navigation. We collect the object goal navigation
data in the HM3D datasets [70] in Habitat simulator [75]. Here,
the agent is initialized in unseen environments and is required
to find one object that could be a category set of chair, couch,
TV, toilet, bed, and plant. We deploy L3MVN [102] in the
HM3D training split and collect the navigation trajectory. We
only collect 483 k navigation trajectory which is successful
in finding the target. And we use the instruction template:
”Search for a/an [Object].” Note that, using the shortest path
to the target object as the navigation video samples leads to
an extremely low performance (30.1% SR) because the model
can not learn to explore or recover from mistakes.
Embodied question answering. We collect 240 k video
action samples from MP3D-EQA dataset [21] and 10 k video
answering samples. The video-action samples are rendered
based on the GT trajectory of MP3D-EQA, and use the
question as the instruction. And the video-answering samples
are rendered frames of full trajectory, and we use the GT
answering as the answer of large language model of MP3D
EQA.
Human following. We collect 544 k video-action samples
from a self-build human following environment based on the
Habitat 3.0 simulator. Specifically, we generate a large amount
of human-following data based on the HM3D dataset [70]. For
each episode, we first determine the total number of humans
(2–6) based on the area of the scene, assign an ID to the target
human to be followed, and specify the language instruction.
Next, we randomly initialize the starting positions of the
humans in the scene and place the robot near the target human,
ensuring that the target is visible in the robot’s camera view.
Finally, we randomly generate multiple navigation waypoints
for each human and use a low-level path planner to guide them
to these waypoints sequentially. During this process, the robot
RealSense D455
LiDAR-L1
Fig. 10: Robot setup. We use Unitree GO2 as our embodiment,
and we mount RealSense D455, a portable Wi-Fi and a LiDAR
L1. Note that, our model only takes RGB frames as input. The
portable Wi-Fi is used for communication with the remote
server and the Lidar is used for the local controller API of
Unitree Dog.
continuously receives the target human’s position in each step
and uses a local planner to follow the human while avoiding
obstacles nearby until the human reaches the final waypoint.
B. Instruction Augmentation
We collect instructions from various datasets, and the quality
of the instructions is limited in diversity. Especially for object
goal navigation, the fixed template could cause the instruction to
become a task indicator, which could damage the performance.
In this case, we use ChatGPT, to augment the instructions. The
prompts are listed as follows:
Given a robot navigation task instruction, rephrase
the instruction to make its grammar and structure
more diverse while preserving its original meaning.
Additionally, ensure all grammatical errors in the
instruction are corrected. Use varied sentence structures
and descriptions for objects and directions to enhance
linguistic diversity. Keep the instructions clear and
concise to ensure they remain suitable for a robot
navigation system.
We find the instruction augmentation could increase the
performance of VLN (+2.31% in SR) and ObjectNav(+3.7%
in SR). We believe the cross-dataset instruction augmentation
could be a promising topic to further investigate.
XII. REAL-WORLD DEPLOYMENT
A. Robot Setup.
We provide a visualization of our robotic dog in Fig. 10.
Our robotic dog is Unitree GO2 and we mount a RealSense
D455 camera on the head of the robotic dog. Here, we only
use the RGB frames with a resolution of 640 × 480 in the
setting of 90◦ HFOV. We also mount a portable Wi-Fi at the
back of the robot dog, which is used to communicate with the
Instruction
Command
Actions
performance of understanding the marionettes with egocentric
Server-A100
(Uni-NaVid)
Fig. 11: Real-world system architecture. We deploy our
model at a remote server, and the robot communicates with
the server through the Internet.
remote server (send captured images and receive commands).
Unitree GO2 is integrated with a LiDAR-L1, which is only
used for local motion planning.
Note that Uni-NaVid does not rely on any odometry
algorithms [108, 109] or noiseless depth [76, 115, 48], making
it easy to deploy in real-world environments.
B. Real-world System Architecture
We provide a visualization of the real-world setup in Fig. 11.
Our model is deployed on a remote server equipped with an
NVIDIA A100 GPU. During navigation, the server receives
navigation instructions and images captured by the robotic
dog through the Internet. To ensure efficient communica
tion, the images are compressed before transmission. After
processing the newly captured images, the model generates
action commands (FORWARD, LEFT, RIGHT, and STOP)
and sends them to the robotic dog. Upon receiving these
commands, the robotic dog executes the actions using a local
motion planning model (specifically, the off-the-shelf model
provided by Unitree Dog). Leveraging our online token merging
strategy, the model processes newly captured images efficiently,
requiring approximately 0.2 seconds for inference, while
Internet communication (image transmission and command
reception) takes about 0.3 seconds.
Non-blocking navigation. After receiving a set of four action
commands, the robotic dog executes them sequentially. Upon
completing each action, the robot captures a new image and
transmits it to the server. In cases where multiple action
commands are generated for a single step, the robot prioritizes
and executes the most recent command, as it reflects the latest
planning outcome.
XIII. EXPERIMENTS
A. Experiment Setup
1) Benchmark.: We conduct extensive experiments on public
benchmarks across different tasks. We celebrate them as
follows: Vision-and-language navigation. We conduct exper
iments on the Val-Unseen split of VLN-CE R2R (Main
paper Table 2) and RxR (Main paper Table 3). They contain
novel environments and novel instructions. Embodied Question
answering. We conduct experiments on the validation set of
the MP3D-EQA [90], where the questions and scenes are
novel. Besides, we also conduct experiments (Table XIII)
on OpenEQA [63], which is a benchmark to evaluate the
video. Human following. In Table 5 of the main paper, the
Perception
Perception
experiment is conducted on a self-build benchmark based on
Environments
the HM3D dataset, which also served as the source of the
training dataset. Additionally, we also build two benchmarks
based on the HSSD dataset and MP3D dataset, respectively, and
evaluate the human following capability of Uni-NaVid across
different datasets and scenes (Table XIV).
2) Metrics.: We use the default metrics of each benchmark.
For vision-and-language navigation and object goal naviga
tion, [44, 76, 94], we use SR, OSR, and SPL as metrics.
Specifically, SR (Success Rate) measures the proportion of
tasks in which the agent successfully reaches the target location
(in VLN the distance threshold is 3 m, and in ObjectNav is
1 m) within the allowed time steps (Up to 500 steps). OSR
(Oracle Success Rate) extends SR by accounting for the agent’s
proximity to the goal, considering the task successful if the
agent is close enough, even without directly stopping at the
target. SPL (Success weighted by Path Length) evaluates the
agent’s efficiency by combining success with the optimality
of its trajectory, penalizing longer-than-necessary paths while
rewarding those closer to the shortest possible route.
For human following navigation, we use SR, FR, and CR
metrics. SR (Success Rate) measures the proportion of events
in which the agent successfully follows the target human to
the endpoint. FR (Following Rate) refers to the proportion of
steps in which the agent successfully follows the target human
for each step. CR (Collision Rate) refers to the proportion of
episodes in which the agent collides with the human during
the movement.
For embodied question answering, we use ACC metric,
which directly measures the percentage of correct answers.
For ScnaQA [7], we use EM, BLUE, ROUGE, METEOR,
CIDEr as metrics. Specifically, EM (Exact Match) evaluates
the percentage of predictions that exactly match the reference
answers. BLEU (Bilingual Evaluation Understudy) measures
the n-gram overlap between the generated text and the reference,
rewarding precise matches in shorter sequences. ROUGE
(Recall-Oriented Understudy for Gisting Evaluation) assesses
the overlap of word sequences, focusing on recall and capturing
the informativeness of the generated text. METEOR (Metric for
Evaluation of Translation with Explicit ORdering) combines
precision and recall, using synonym matching and stemming to
account for variations in phrasing. Lastly, CIDEr (Consensus
based Image Description Evaluation) measures the similarity of
generated responses to multiple references by capturing human
consensus, emphasizing relevance and diversity in the output.
For video-question answering benchmark MSVD-QA [95],
MSRVTT-QA [95], and ActivityNet-QA [104], we use ACC
and Score metrics. Specifically, ACC (Accuracy) measures
the proportion of correctly answered questions, providing a
straightforward assessment of the model’s overall correctness.
Score, on the other hand, evaluates the quality of the generated
answers using GPT (GPT 3.5 in implementation) as the zero
shot evaluation assistor to assign relative scores on a scale of
1 to 5 for generated answers.
Method SimpleIns. ComplexIns.
NaVid[111] 80% 20%
Uni-NaVid 92% 84%
TABLEXI: Real-worldVLNexperiments.We compare
ourmethodwithNaVidontwotypesof instructions: simple
instructions(25)andcomplexinstructions(25).
Method Observation VLN-CERxRVal-Unseen
Odom.DepthS.RGB TL NE↓OS↑SR↑SPL↑
LAW[73] ✓ ✓ ✓ 4.01 10.8721.0 8.0 8.0
CM2[26] ✓ ✓ ✓ 12.29 8.98 25.3 14.4 9.2
WS-MGMap[16] ✓ ✓ ✓ 10.80 9.83 29.8 15.0 12.1
ETPNav.FF[88] ✓ ✓ ✓- 8.79 36.7 25.5 18.1
Seq2Seq[44] ✓ ✓ 1.16 11.8 5.02 3.51 3.43
CMA[44] ✓ ✓ 5.09 11.7 10.7 4.41 2.47
A2Nav† [17] ✓––– 16.8 6.3
NaVid[111] ✓ 10.59 8.41 34.5 23.8 21.2
Uni-NaVid ✓ 8.3 8.08 40.9 29.5 28.1
TABLEXII:Vision-and-languagenavigation(RxR).Compar
isononVLN-CERxR[47]Val-Unseen. †: indicateszero-shot
methods.
B. Real-worldExperiments
Weconduct real-worldexperiments tostudy thegeneral
izabilityof ourmethod in the realworld. Specifically,we
leverage theVLNtask,whichincludesbothlandmarksand
motions, to evaluate ourmethodwith the previousVLN
methodNaVid [111]. Following previouswork [111], we
designed two types of instructions for different difficulties
(25 simple instructions and25 complex instructions). The
simpleinstructions,whichrequiretheagent tonavigatetoa
singlerobotlandmarkandstop;(2)complexinstructions,which
requiretheagent tofollowaseriesofsimpleinstructions.
Welist someexamplesofusedinstructions inexperiments.
Simpleinstructionexamples:
•Moveforward1step, thenstop.
•Turnleft5steps, thenstop.
•Movetotherightchair, thenturnleft.
Complexinstructions:
•Moveforward5steps, thenturnleft4steps,andfinally
turnright5steps.
•Gostraight tothechair, thenturnrightandmovetothe
door, stopbythedoor.
•Turnalargeright,andgoforwardtoaplant, thenright
at theplantandmovetotheTV, stopclosetoTV.
Herewepresentour results inTableXI.Wefindthatboth
NavidandourmethodcanachievehighSRinsimpleinstruc
tions,However, forcomplexinstructions,ourmethodshows
significant improvements.Thisdemonstrates thesuperiorityof
ourmethodoverexistingmethods.Weaddmorevisual results
totheattachedvideo.
C.MoreBenchmarkExperiments
Cross-datasetVision-and-LanguageNavigationEvaluation.
Weevaluatethecross-datasetperformanceofourmethodon
VLN-CERxRbyexcludingtheVLN-CERxRdataat training
Method
EM-EQA
ScanNet
Eq. (1)
HM3D
Eq. (1)
ALL
Eq. (1)
BlindLLMs
GPT-4 32.5 35.5 33.5
LLaMA-2 27.9 29.0 28.3
SocraticLLMsw/FrameCaptions
GPT-4w/LLaVA-1.5 45.4 40.0 43.6
LLaMA-2w/LLaVA-1.5 39.6 31.1 36.8
SocraticLLMsw/Scene-GraphCaptions
GPT-4w/CG 37.8 34.0 36.5
LLaMA-2w/CG 31.0 24.2 28.7
GPT-4w/SVM 40.9 35.0 38.9
LLaMA-2w/SVM 36.0 30.9 34.3
Multi-FrameVLMs
GPT-4V 51.3 46.6 49.6
HumanAgent 87.7 85.1 86.8
Uni-NaVid 41.4 38.1 40.3
TABLEXIII:Embodiedvideoquestionanswering.Compar
isononOpenEQAbenchmark.EM-EQAresultsarebroken
downbydatasource(ScanNet,HM3D, andALL).GPT-4V
scoresarecalculatedonasubsetof500OpenEQAquestions
duetoAPI limitations.
time.The resultsarepresented inTableXII.Notably, even
without trainingonRxR,ourmethodstilloutperformsexisting
approaches.However,asignificantdecreaseinallmetrics is
observedcompared towhenRxRtrainingdata is included.
Wehypothesizethat thisdiscrepancyarisesfromdifferences
in trajectorycharacteristics: trajectories in theR2Rdataset
have relativelyuniformlengths (approximately10meters),
whereas thoseinRxRexhibitgreaterdiversity, rangingfrom
approximately2meters to20meters.Thisdisparityconstrains
ourmethod’sperformanceontheRxRdatasetandunderscores
theimportanceof trainingondiversetrajectories.
OpenEQAbenchmark[63].Tofurtherevaluateourmethod
onembodiedquestionanswering,weconductexperiments in
OpenEQA,which require themethods toanswer questions
by analyzing an egocentric video. The results are shown
in Tab. XIII. We find our method achieves competitive
performancetothestrongestcommercialusedVision-Language
model suchasGPT-4V.Besides, bydirectlyobservingand
understandingthevideosequences,ourmethoddoesnotneed
additional framecaptionorSocratic-stylepromoted[63]used
inotherLLMs.
Cross-EnvironmentsFollowing.Weevaluateourmethodin
novelenvironments, includingHSSD[42] (syntheticenviron
ments) andMP3D[11] (reconstructed environments). The
results,presentedinTableXIV,demonstratethatourapproach
consistentlyoutperforms baselinemethods inbothSRand
FRmetrics acrossHSSDandMP3D.Notably, ourmethod
achievessignificant improvements inHSSD, likelyduetothe
absenceof reconstructionartifacts insyntheticenvironments,
whichreduces thelikelihoodof therobotbeingstuck.For the
CR(collisionrate)metric, IBVSadoptsahighlyconservative
strategythatmaintainsaconsiderabledistancefromthetarget
(i.e., alargeboundingbox).Whilethisresults inalowerCR,
Method HF-HSSD HF-MP3D
SR↑ FR↑ CR↓ SR↑ FR↑ CR↓
PoliFormer [106] 2.67 20.81 0.97 2.62 16.59 1.43
PoliFormer∗ [106] 26.97 54.20 10.01 25.42 47.80 8.72
PoliFormer†[106] 27.10 53.96 9.34 21.90 41.42 7.15
IBVS∗ [28] 66.32 80.26 0.19 56.86 68.91 1.33
IBVS†[28] 65.36 80.33 0.15 58.15 65.83 0.77
Uni-NaVid 81.65 89.34 1.33 69.80 78.96 2.99
TABLEXIV: Human following. Comparison onHuman
FollowingHSSDBenchmark(HF-HSSD)andHumanFollow
ingMP3DBenchmark(HF-MP3D). ∗:MethodsuseGround
ingDINO [58] as the open-vocabulary human detector. †:
Methodsusetheground-truthboundingboxprovidedbythe
simulator.
Design VLN(SR↑)ObjNav(SR↑)EQA(ACC↑)Follow(SR↑)
4tokensforCurr.Obs. 43.4 50.2 40.5
1tokenforCurr.Obs. 35.1 45.3 31.2
1tokenforShort.Obs. 46.2 69.2 44.9 60.3
τ=0.9 40.2 70.0 43.2 57.8
τ=0.95 48.7 73.7 47.3 61.2
τ=0.99 49.6 70.2 48.1 57.2
TABLEXV:Ablationstudyof tokennumberandτ
itadverselyaffectsFRandSRperformance.
D.MoreAblationstudy
We conduct additional ablation studies to validate the
effectivenessofourkeydesigncomponents.Theresultsare
presented inTableXV.Notably,weobserveda significant
performancedropwhen thenumber of current observation
tokenswas reduced. Inparticular, the following taskfailed
acrossallsequences,potentiallyduetotheinabilityofasmaller
numberof tokenstoprovidesufficient informationfor tracking
humanmotion.Furthermore,wefoundthatincreasingthevalue
ofτ ledtoimprovedperformance.Thisoutcomeis intuitive,
asahigherτ retainsmorelong-termobservationtokens,which
arecrucial forunderstandingnavigationhistory.
E. TimeAnalyze
Toevaluate theefficiencyofourmethod,wepresent the
average running timeand tokencount acrossdifferent time
steps inFig.12.Ourapproachiscomparedagainst existing
models thatalsoemploytokenmergingstrategies.Theresults
Number of Frames Number of Frames
Time (s)
Number of Visual Tokens
Uni-NaVid LLaMA-VID NaVid
Fig.12:Timeefficiencyvisualization.Weprovidetheaverage
runningtimeandthenumberof tokensacrossdifferent time
steps.
demonstratethatourmethodismoreefficientwhichachieves
an inference timeof approximately0.2seconds.Moreover,
ourmethodmaintainsconsistent runningtimes,whereas the
runningtimesofexistingmethodsincreasesignificantly.These
advantages stem from two key factors: First, our model
architectureishighlyefficient,whereasotherapproaches, such
asLLaMA-VID[51]andNaVid[111],relyontime-consuming
operations likeQFormer. Second, ouronline tokenmerging
strategyresults inagradual increaseintokencount, ensuring
morestableinferencetimes.
F. QualitativeExperiments
Weprovidevisual resultsofourmethodonvariousbench
marks:Fig.13illustratesresultsforVLN-CER2RandRxR,
Fig.14forHM3DObjectNav, andFig. 15forMP3D-EQA,
Fig. 16 forHM3Dhuman following. For additional visual
results,pleaserefer totheattachedvideo.
Vision-and-language Navigation
VLN-CE R2R I
Videos
Tirid-Person View
Walk out of the sauna into the pool room and take a left. Walk towards the bar and take a left going up the stairs into the hallway. 
Go up the next part of the stairs and take at the first right into the massage room. Stop just inside the doorway on the right.
Steps
VLN-CE R2R II
Videos
Tirid-Person View
Turn left and walk forward towards the white tub. Turn right towards the first room that is next to the white tub, not on the
corner. Walk inside the room. There should be a toilet bowl visible, and stop there.
Steps
VLN-CE RxR I
Videos
Tirid-Person View
You‘re standing in front of the frame of a door, turn to your left and you will notice an open door that leads to a bedroom, walk 
towards the foot of the bed in front of you in that room, and then turn left. You will see there’s a mirror on your right side, and 
a picture hanging on the left wall. There‘s an open door in front of you, walk out of the room and head towards an open door 
on your left which leads to another bedroom. Once you’re in the bedroom you will see a bed with blue sheets on your right, 
and an open door your left, walk past the open door on your left and as you Hook to your left in you will see a second open 
door that leads to a room. Walk into the bathroom and stand in front of the sink. That‘s your spot.
Steps
VLN-CE RxR II
Videos
Tirid-Person View
You are facing towards the glass window. Turn slightly right and exit the room. Then move forward, until you see an opening 
glass door. Enter into that room where it has a glass table in front of you and some wall hanging on the right, then take left and 
stop facing towards the blackboard, and this is your endpoint.
Steps
Fig. 13: Visual results of VLN on VLN-CE R2R and RxR.
Object goal navigation
Videos
HM3D I
Tirid-Person View
Search for a toilet.
Steps
Videos
HM3D II
Tirid-Person View
Search for a bed.
Steps
Videos
HM3D III
Tirid-Person View
Search for a chair.
Steps
Videos
HM3D IV
Tirid-Person View
Search for a bed.
Steps
Fig. 14: Visual results of HM3D ObjectNav.
Embodied Question Answering
MP3D-EQA I
Videos
Tirid-Person View
What color is the stove? Uni-NaVid: Sliver
Steps
MP3D-EQA II
Videos
Tirid-Person View
What color is the wardrobe? Uni-NaVid: Brown
Steps
Videos
MP3D-EQA III
Tirid-Person View
What room is the refrigerator located in? Uni-NaVid: Kitchen
Steps
Videos
MP3D-EQA III
Tirid-Person View
What color is the sofa in the lounge? Uni-NaVid: Brown
Steps
Fig. 15: Visual results of MP3D-EQA.
Human Following
Human Following HM3D
Videos
Follow the woman wearing a yellow shirt and dark shorts.
Steps
Human Following HM3D
Videos
Follow the person dressed in yellow and dark..
Steps
Human Following HM3D
Videos
Follow the woman in gray shorts.
Steps
Human Following HM3D
Videos
Follow the man wearing a blue t-shirt and gray trousers.
Steps
Human Following HM3D
Videos
Follow the person in yellow.
Steps
Fig. 16: Visual results of HM3D Human following.




offline_eval.py
# coding: utf-8
import os
import json
import cv2
import numpy as np
import imageio
import json
import torch
import cv2
import time
import argparse

from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria




seed = 30
torch.manual_seed(seed)
np.random.seed(seed)


class UniNaVid_Agent():
    def __init__(self, model_path):
        
        print("Initialize UniNaVid")
        
        self.conv_mode = "vicuna_v1"

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))

        assert self.image_processor is not None

        print("Initialization Complete")
        
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop."
        self.rgb_list = []
        self.count_id = 0
        self.reset()

    def process_images(self, rgb_list):

        
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()

        return [video]


    def predict_inference(self, prompt):
        
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        self.rgb_list = []

        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.5,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs




    def reset(self, task_type='vln'):

        self.transformation_list = []
        self.rgb_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []
        self.task_type = task_type

        self.first_forward = False
        self.executed_steps = 0
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0


    def act(self, data):
    
        rgb = data["observations"]
        self.rgb_list.append(rgb)


        navigation_qs = self.promt_template.format(data["instruction"])
        
        navigation = self.predict_inference(navigation_qs)
                
        action_list = navigation.split(" ")

        traj = [[0.0, 0.0, 0.0]]
        for action in action_list: 
            if action == "stop":
                waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, 0.0])]
                traj = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                break
            elif action == "forward":
                waypoint = [x + y for x, y in zip(traj[-1], [0.5, 0.0, 0.0])]
                traj.append(waypoint)
            elif action == "left":
                waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, -np.deg2rad(30)])]
                traj.append(waypoint)
            elif action == "right":
                waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, np.deg2rad(30)])]
                traj.append(waypoint)

                                    
        if len(action_list)==0:
            raise ValueError("No action found in the output")
            
        self.executed_steps += 1
            
        self.latest_action = {"step": self.executed_steps, "path":[traj], "actions":action_list}
            
        return self.latest_action.copy()

def get_sorted_images(recording_dir):
    image_dir = os.path.join(recording_dir, 'images')
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    images = []
    for step, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        np_image = cv2.imread(image_path)
        images.append(np_image)
    
    return images

def get_traj_data(recording_dir):
    json_path = os.path.join(recording_dir, "instruction.json")

    with open(json_path, 'r', encoding='utf-8') as f:
        instruction = json.load(f)["instruction"]

    return instruction

def draw_traj_arrows_fpv(
    img,
    actions,
    arrow_len=10,                
    arrow_gap=2,                 
    arrow_color=(0, 255, 0),    
    arrow_thickness=2,
    tipLength=0.35,
    stop_color=(0, 0, 255),      
    stop_radius=5
):
 
    out = img.copy()
    h, w = out.shape[:2]

    base_x, base_y = w // 2, int(h * 0.95)

    for i, action in enumerate(actions):
        if action == "stop":
            waypoint = [0.0, 0.0, 0.0]
        elif action == "forward":
            waypoint = [0.5, 0.0, 0.0]
        elif action == "left":
            waypoint = [0.0, 0.0, -np.deg2rad(30)]
        elif action == "right":
            waypoint = [0.0, 0.0, np.deg2rad(30)]
        else:
            continue  

        x, y, yaw = waypoint

        start = (
            int(base_x),
            int(base_y - i * (arrow_len + arrow_gap))
        )

        if action == "stop":
            cv2.circle(out, start, stop_radius, stop_color, 2)
        else:
            end = (
                int(start[0] + arrow_len * np.sin(yaw)),
                int(start[1] - arrow_len * np.cos(yaw))
            )
            cv2.arrowedLine(out, start, end, arrow_color, arrow_thickness, tipLength=tipLength)
    
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out





if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('test_case', help='test case path (images dir)')
    parser.add_argument('output_dir', nargs='?', default='output_dir', help='output dir to save results (default: output_dir)')
    

    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist (convert to absolute path)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    agent = UniNaVid_Agent("./model_zoo")
    agent.reset()
     
    images = get_sorted_images(args.test_case)
    instruction = get_traj_data(args.test_case)
    print(f"Total {len(images)} images")
    h,w,n = images[0].shape
        
    result_vis_list = []
    step_count = 0
    for i, img in enumerate(images):
        image=img

        import time
        t_s = time.time()
        result = agent.act({'instruction': instruction, 'observations': image})
        step_count += 1
        
        print("step", step_count, "inference time", time.time()-t_s)
        
        traj = result['path'][0]
        actions = result['actions']

        vis = draw_traj_arrows_fpv(img, actions, arrow_len=20)
        result_vis_list.append(vis)

    
    # Ensure output directory exists before saving
    os.makedirs(output_dir, exist_ok=True)
    imageio.mimsave(os.path.join(output_dir, "result.gif"), result_vis_list)
```
