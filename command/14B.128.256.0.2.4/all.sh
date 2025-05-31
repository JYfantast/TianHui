#!/bin/bash

#Qwen2.5-Math-14B PT
llamafactory-cli train command/14B.128.256.0.2.4/PT.yaml
#Qwen2.5-Math-14B PT merge
llamafactory-cli export command/14B.128.256.0.2.4/PT.merge.yaml
##Qwen2.5-Math-14B SFT
llamafactory-cli train command/14B.128.256.0.2.4/SFT.yaml
#Qwen2.5-Math-14B PT merge
llamafactory-cli export command/14B.128.256.0.2.4/SFT.merge.yaml
