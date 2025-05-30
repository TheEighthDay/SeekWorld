<div align="center">

<div align="center">
    <img alt="SeekWorld logo" src="./assets/seekworld2.jpg" style="height: 140px;" />
</div>

<h1>SeekWorld: Geolocation is a Natural RL Task for o3-like Visual Clue-Tracking Reasoning</h1>


<div align="center">
  <!-- <a href="xxx"><img src="https://img.shields.io/static/v1?label=Arxiv&message=SeekWorld&color=red&logo=arxiv"></a> &ensp; -->
  <a href="https://huggingface.co/datasets/TheEighthDay/SeekWorld"><img src="https://img.shields.io/static/v1?label=Dataset&message=SeekWorld&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/TheEighthDay/SeekWorld_RL_PLUS"><img src="https://img.shields.io/static/v1?label=Model&message=SeekWorld-7B&color=blue"></a> &ensp;
  <a href="https://huggingface.co/spaces/TheEighthDay/SeekWorld_APP"><img src="https://img.shields.io/static/v1?label=Demo&message=Online&color=red"></a> &ensp;
</div>

</div>

<hr>

[切换到英文版 (Switch to English version)](/README.md)

## News

- [2025/4/20] ✨ **第一个尝试复现o3-like的视觉线索跟踪推理能力的项目！** 我们开源了**SeekWorld**的数据集与直接RL训练的模型**SeekWorld-7B**! 

## 👀 About SeekWorld

为了提高多模态大语言模型(MLLMs)的性能，近期一些方法尝试通过图像数学题、图表分析和逻辑谜题等任务来激发模型的纯推理能力；也有一些方法聚焦于通过传统视觉检测任务（如目标检测、计数、分割）来增强模型的低层感知能力。此外，还有研究致力于在推理过程中以文本形式重新感知视觉内容。然而，一个关键的局限在于：**当前的MLLM在进行视觉推理时仍完全依赖于纯文本信息**。

OpenAI的 **[ChatGPT-o3](https://openai.com/index/thinking-with-images/)** 实现了基于思维链的视觉推理，允许模型在推理过程中动态操作图像（如旋转、缩放、变换等）。例如，“but I’ll zoom in a bit just to be absolutely sure!” 这一表达体现了其交互式的推理能力，极大提高了推理过程中的感知能力，使其能够挖掘细致、模糊或容易被忽视的视觉线索，构建了一条连贯的视觉推理证据链。其中官方有一个有趣示例是通过一张图片定位到图片拍摄地区曾经拍摄过的电影，在这样的场景中我们需要挖掘视觉线索——推理——挖掘视觉线索——再推理...直到得出最终结果。因此，我们认为“Visual Clue-Tracking”是对这一能力的贴切概括。

因此我们提出了一项新任务：**地理定位推理（Geolocation Reasoning）**。该任务要求模型在感知视觉信息的同时，推理出图像中视觉语义所隐含的高级逻辑关系，从而确定图像的拍摄地点，极其适合用于实现类似 o3 的视觉线索跟踪推理。你可以通过以下两个“猜图地点”类游戏更实际感受这一任务：[GeoGuess](https://www.geoguessr.com/) 和 [TuXun](https://tuxun.fun/)。围绕该任务我们构建了一个基于规则的地理定位强化学习数据集：**[SeekWorld](https://huggingface.co/datasets/TheEighthDay/SeekWorld)**。

该数据集包含两个训练集，其中一个（**Train-Clue-Tracking**）包含**50条从o3中收集的针对视觉线索跟踪的详细推理过程数据（数据持续扩充中）**，另一个（**Train-No-Process**）则包含**8541条不含推理过程的普通样本数据**。前者用于模型 **Cold-Start 阶段的 SFT 训练**，后者则用于 **RL 训练**。我们还提供了两个测试集，用于综合评估模型的性能。

目前我们已基于 Train-No-Process 数据，并以 Qwen2.5-7B-VL-Instruct 为基础模型，通过强化学习训练得到一个专门的视觉地理定位模型：**[SeekWorld-7B](https://huggingface.co/TheEighthDay/SeekWorld_RL_PLUS)**。


## 🌟 Future Work
- [ ] **继续扩充Cold-Start SFT数据集的规模**
- [ ] **Cold-Start SFT (Train-Clue-Tracking) + RL (Train-No-Process)** ：目前还未完成Cold-Start SFT的训练，进而复现出o3-like的视觉线索跟踪能力。
- [ ] **评估o3在SeekWorld上的效果** ：由于API限制目前还未能在SeekWorld上评估o3的效果。
- [ ] **评估不同感知与推理benchmark的效果** ：评测通过地理定位推理训练过后的o3-like模型在其他领域的效果。

## 🔍 Dataset
* **包含视觉推理过程**: 第一个包含o3模型视觉思维链或者视觉线索跟踪能力的数据集。
* **全球多样化采样**：涵盖了来自世界各地广泛的场景集合，确保模型能够对多样的文化、地形和背景环境实现强大的泛化能力。
* **针对rule-based RL优化的图像-标签对**：对于图片清洗了包含位置信息的水印，对于地理坐标标签增加了一级行政规划区的别名，防止模型被错误惩罚。
* **分层难度架构**：包含三个不同的推理难度层级——简单、中等和困难，以逐步挑战和评估模型在地理定位方面的能力。


<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Data Volume</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
     <tr>
      <td>Train-Clue-Tracking</td>
      <td>50(持续扩充中)</td>
      <td>Collection of visual clue-tracking reasoning process from o3</td>
    </tr>
    <tr>
      <td>Train-No-Process <br>(easy-medium-hard)</td>
      <td>8541  <br>(1945-941-5655)</td>
      <td>Panoramas  & user-uploaded images from Google Maps in recent years</td>
    </tr>
    <tr>
      <td>Global-Test</td>
      <td>320</td>
      <td>Panoramas  & user-uploaded images from Google Maps in recent years</td>
    </tr>
    <tr>
      <td>China-Test</td>
      <td>373</td>
      <td>The latest Xiaohongshu App images collected on April 14, 2025, and it is almost impossible to have been pre-trained</td>
    </tr>
  </tbody>
</table>

Google Driver: [SeekWorld](https://drive.google.com/drive/folders/115X73SRULCYLKZqd3UHs4MIG3PI4BSkw?usp=sharing).  有关数据集的更多细节 [DATASET.md](DATASET.md).

<img src="assets/dataset.png" width="1000px"/>

## 🏆 Performance
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Global-Test</th>
            <th>China-Test</th>
            <th>Overall Accuracy</th>
        </tr>
    </thead>
    <tbody>
        <!-- 第一组标题 -->
        <tr>
            <td colspan="4" style="text-align:center; font-style:italic;"><em>Bigger model</em></td>
        </tr>
        <tr>
            <td><a href="https://openai.com/index/gpt-4o-system-card/">GPT4o-240806</a>🔒</td>
            <td><b>56.50</b></td>
            <td>31.90</td>
            <td><b>43.26</b></td>
        </tr>
        <tr>
            <td><a href="https://team.doubao.com/zh/special/doubao_1_5_pro">Doubao-1.5-vision-pro-32k-250115</a>🔒</td>
            <td>43.75</td>
            <td><b>40.48</b></td>
            <td>41.99</td>
        </tr>
        <tr>
            <td><a href="https://deepmind.google/technologies/gemini/flash-thinking/">Gemini-2.0-flash-thinking-exp-01-21</a>🔒🧠</td>
            <td>56.25</td>
            <td>29.49</td>
            <td>41.85</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/Qwen/QVQ-72B-Preview">QvQ-72B-max-2025-03-25</a>🧠</td>
            <td>48.13</td>
            <td>31.63</td>
            <td>39.25</td>
        </tr>
        <tr>
            <td><a href="https://github.com/QwenLM/Qwen2.5-VL">Qwen-2.5-32B-VL</a></td>
            <td>38.12</td>
            <td>24.13</td>
            <td>30.59</td>
        </tr>
        <!-- 第二组标题 -->
        <tr>
            <td colspan="4" style="text-align:center; font-style:italic;"><em>Small model (7B)</em></td>
        </tr>
        <tr>
            <td> <b><a href="https://huggingface.co/TheEighthDay/SeekWorld_RL_PLUS">SeekWorld-7B</a></b> [Cold-Start SFT + RL] <i>(ours)</i></td>
            <td><b>-</b></td>
            <td><b>-</b></td>
            <td><b>-</b></td>
        </tr>
        <tr>
            <td> <b><a href="https://huggingface.co/TheEighthDay/SeekWorld_RL_PLUS">SeekWorld-7B</a></b> [Direct RL] <i>(ours)</i></td>
            <td><b>59.69</b></td>
            <td><b>34.65</b></td>
            <td><b>46.21</b></td>
        </tr>
        <tr>
            <td><a href="https://github.com/QwenLM/Qwen2.5-VL">Qwen-2.5-7B-VL</a> [Direct RL] </td>
            <td>51.25</td>
            <td>31.90</td>
            <td>40.84</td>
        </tr>
        <tr>
            <td><a href="https://github.com/QwenLM/Qwen2.5-VL">Qwen-2.5-7B-VL</a> [Direct SFT] </td>
            <td>37.19</td>
            <td>25.47</td>
            <td>30.88</td>
        </tr>
        <tr>
            <td><a href="https://github.com/QwenLM/Qwen2.5-VL">Qwen-2.5-7B-VL</a></td>
            <td>33.44</td>
            <td>24.40</td>
            <td>28.57</td>
        </tr>
        <tr>
            <td><a href="https://github.com/QwenLM/Qwen2.5-VL">Qwen-2.5-7B-VL</a> + <a href="https://arxiv.org/pdf/2502.13759">CoT</a></td>
            <td>25.31</td>
            <td>21.45</td>
            <td>23.23</td>
        </tr>
    </tbody>
</table>


带有🔒标识的模型是**专有闭源模型**，而带有🧠标识的模型则**具备增强的推理能力**。我们采用(<a href="https://arxiv.org/pdf/2501.03262">Reinforce++</a>)作为RL算法。

我们目前尚未完成在Train-Clue-Tracking上的Cold-Start SFT训练，Direct SFT和Direct RL分别指在Train-No-Process上直接进行SFT和RL训练。相比于Qwen-2.5-7B-VL，SeekWorld-7B在RL训练中尝试了两项优化。其中，难度采样有效地提高了测试的准确率。长度激励仅增加了推理过程的长度，而没有提高准确率。不过幸运的是，它能更好地展示中间推理过程。我们也在尝试使用GRM([code for geolocation reasoning](./src/lmm-r1/openrlhf/models/remote_rm/location_verifier_process.py)).

* **难度采样**：我们在训练集中对不同难度级别的问题进行了采样。具体来说，由于数据集中难题数量相对较多（5655道），我们将简单题（1945道）和中等难度题（941道）的数量翻倍，分别扩充到3890道和1882道。

* **长度激励**：我们尝试引入了一种鼓励增加推理长度的奖励机制。奖励值设定为字符数量乘以0.001 ，奖励值最高为1.0。 

## 🖥️ Demo
让我们测试一张图片感受地理定位的魅力来自于o3：

猜猜这是哪里？

![](./assets/chongqing.jpg)

视觉线索跟踪过程和结果在[o3_demo.md](/o3_demo.md).

再看看我们SeekWorld-7B的效果：
```bash
CUDA_VISIBLE_DEVICES=0 python simple_inference.py --image_path test.jpg --model_name TheEighthDay/SeekWorld_RL_PLUS --inference_engine transformers
```
<img src="./assets/test.jpg" width="300px"/>

你可以在我们的在线demo上尝试 [Online Demo](https://huggingface.co/spaces/TheEighthDay/SeekWorld_APP)~

```text
<思考>
这张图片描绘了一个传统的中国街道场景，有木质建筑、红灯笼以及墙上的汉字。建筑的结构和风格，再加上红灯笼和文字等中国传统元素的存在，表明这是中国一个村庄或城镇的场景。这些建筑的设计和布局具有中国古代城镇的特点，这些城镇往往具有历史意义，并且以其保存完好的传统建筑而闻名。
汉字的存在以及建筑风格表明，这里很可能在中国，具体来说是在一个传统建筑较为常见的地区。建筑风格和整体氛围可能暗示着像丽江这样的地方，丽江以其保存完好的古城而闻名，它位于云南省，该省以其丰富多样的文化遗产和历史遗迹而著称。
鉴于其建筑风格和文化元素，合理推断这张图片来自中国一个有着丰富历史和传统建筑的地区。建筑的独特风格以及中国传统元素的存在表明，这里可能是云南省的一部分，云南省以其历经岁月保存下来的古镇和村庄而闻名。
因此，基于建筑风格和文化元素，合理得出结论：这张图片拍摄于中国，具体来说是在中国的一级行政区之一 —— 云南省。
</ 思考 >
< 答案 >
中国，云南
</ 答案>
```
这里正是云南的沙溪古镇，模型给出了详细推理的流程，并定位到了云南~

## 🔥 Training
请参考 [LMM-R1](./src/lmm-r1) 进行训练环境的准备. 
```bash
cd src/lmm-r1/
bash examples/scripts/lmm_r1/train_direct_rl_seekworld.sh
```

## 📈 Dirct RL Training Curves

<img src="./assets/training_curve.png" width="750px"/>

## 🤝 Acknowledgements

我们非常感激: [lmm-r1](https://github.com/TideDra/lmm-r1)和 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)给我提供了优秀的Baseline代码！

## ✏️ Citations

如果您发现我们的工作对您的研究有帮助，请考虑引用我们：
```bibtex
@misc{seekworld2025,
  title = {{S}eek{W}orld: Geolocation is a Natural {RL} Task for o3-like Visual Clue-Tracking},
  author = {Tian, Kaibin and Xin, Zijie and Liu, Jiazhen},
  year = {2025},
  howpublished = {\url{https://github.com/TheEighthDay/SeekWorld}},
  note = {GitHub repository}
}
```
## 📮 Contribute via Crowdsourcing
我们热烈欢迎参与到 SeekWorld 项目中来！如果您对地理定位推理感兴趣，您可以向我们发送一张具有挑战性的测试图片，以此来帮助我们构建一个更全面的评估数据集。 贡献方式如下：

* 拍摄一张带有地理线索但又不是很容易就能识别出位置的照片（例如，街景、生活照片、建筑、自然景观）。
* 确保该图片对应的是一个真实的地点（例如，具体到国家和一级行政区）。如果可能的话，请同时提供该地点的经纬度。并确保图片中不包含任何个人信息。
* 请在邮件主题中注明：[SeekWorld Crowd Contribution]。然后将图片发送至我们的邮箱地址：**tikibi001@163.com** 。

## 📬 Contact 
Kaibin Tian: 1109419614@qq.com

欢迎在微信中与我们交流:

<img src="./assets/wechat.JPG" width="300px"/>

