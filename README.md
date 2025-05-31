### PT and SFT

```shell
First, refer to LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory) to install the environment required for PT and SFT. 

Next, run command/14B.128.256.0.2.4/14B.0.2.4.all.sh to complete PT, merging, SFT, and merging. 
```

### Answer generation from LLMs

```shell
First, configure the respective environments according to the official documentation of each LLM. 

Next, execute the LLM generation command, such as scripts/14B.128.256.0.2.4.sh, to generate text outputs for the 12 downstream tasks.
```

### Evaluation

```shell
First, set up the required evaluation environment according to Baichuan2 (https://github.com/baichuan-inc/Baichuan2). 

Next, perform the evaluation by running the corresponding downstream task commands, such as scripts.result/QA.sh.
```

## Acknowledgments

1. We are very grateful to Sichuan Huixin Intelligent Computing Technology Co., Ltd. for providing the equipment and technical support (8 NVIDIA A100 40G GPUs) for this research.
2. We sincerely appreciate the open-source projects LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory), TCMChat (https://github.com/ZJUFanLab/TCMChat), and Baichuan2 (https://github.com/baichuan-inc/Baichuan2) for their technical support, which was instrumental in the successful completion of this research.

