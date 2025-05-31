### PT and SFT

During the model training phase, we implemented a two-stage process consisting of PT and SFT within the LLaMA-Factory architectural framework. For the PT phase, the model performs unsupervised learning on large-scale textual data to capture deep semantic information. Specifically, we used DeepSpeed Stage 2 to achieve efficient distributed training and combined Flash Attention 2 technology to accelerate the computation of attention mechanisms, significantly improving training efficiency and model convergence speed. In addition, to address the memory bottleneck problem in large-scale model training, we introduced QLoRA technology, which significantly reduced memory usage and computational requirements by quantifying model weights, enabling efficient PT even with limited hardware resources. In the SFT stage, we used a high-quality annotated dataset for supervised training of the model to further improve its performance on specific tasks. Similar to the PT stage, we also adopted DeepSpeed Stage 2 and Flash Attention 2 techniques in the SFT stage to ensure the efficiency and stability of the training process. Specifically, to further enhance the adaptability and generalization ability of the model, we used instruction data to enhance the adaptability of the deep learning model. Through the comprehensive application of these technologies, we can not only ensure model performance while significantly reducing training time, but also effectively reduce the consumption of computing resources, thereby achieving efficient training of large-scale deep learning models. 

```shell
First, refer to LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory) to install the environment required for PT and SFT. 

Next, run command/14B.128.256.0.2.4/14B.0.2.4.all.sh to complete PT, merging, SFT, and merging. 
```

### Answer generation from LLMs

We conducted a comprehensive evaluation of 12 downstream tasks in TCM using TianHui and other LLMs, encompassing 27 different LLMs and 11 ablation variants of TianHui. The evaluated tasks include: Answer Prediction Question (APQ), TCM Case Diagnosis (TCMCD), TCM Entity Extraction (TCMEE), Herb or Formula Recommendation (HFR), Acupuncture Point Recommendation (APR), Herbal Chemical Composition Analysis (HCCA), Generation of Chinese Patent Medicine Instruction (GCPMI), Description of Herbal Pharmacological Effect (DHPE), TCM Knowledge Question Answering (TCMKQA), TCM Reading Comprehension (TCMRC), Topic-led Abstract Writing (TLAW), and Abstract-driven Topic Generation (ADTG). 

```shell
First, configure the respective environments according to the official documentation of each LLM. 

Next, execute the LLM generation command, such as scripts/14B.128.256.0.2.4.sh, to generate text outputs for the 12 downstream tasks.
```

### Evaluation

We evaluated TianHui using 12 different types (APQ, TCMCD, TCMEE, HFR, APR, HCCA, GCPMI, DHPE, TCMKQA, TCMRC, TLAW, and ADTG) of benchmark test datasets and conducted extensive comparison experiments and ablation experiments. 

```shell
First, set up the required evaluation environment according to Baichuan2 (https://github.com/baichuan-inc/Baichuan2). 

Next, perform the evaluation by running the corresponding downstream task commands, such as scripts.result/QA.sh.
```

## Acknowledgments

1. We are very grateful to Sichuan Huixin Intelligent Computing Technology Co., Ltd. for providing the equipment and technical support (8 NVIDIA A100 40G GPUs) for this research.
2. We sincerely appreciate the open-source projects LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory), TCMChat (https://github.com/ZJUFanLab/TCMChat), and Baichuan2 (https://github.com/baichuan-inc/Baichuan2) for their technical support, which was instrumental in the successful completion of this research.

