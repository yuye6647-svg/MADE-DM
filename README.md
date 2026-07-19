# MADE-DM
# Tackling Incomplete Data via Missing-Aware Dynamic Enhancement of Dominant Modality for Robust Multimodal Sentiment Analysis

Pytorch implementation of the paper:

Tackling Incomplete Data via Missing-Aware Dynamic Enhancement of Dominant Modality for Robust Multimodal Sentiment Analysis

This is a reorganized code, if you find any bugs please contact me. Thanks.

## Data Preparation

MOSI/MOSEI/CH-SIMS Download: Please see [MMSA](https://github.com/thuiar/MMSA)

## Environment

The basic training environment for the results in the paper is Python 3.9.7, PyTorch 2.2.1 with an NVIDIA 3090 GPU (24 GB memory).

## Note
This work builds upon our previous work [LNLN](https://github.com/Haoyu-ha/LNLN), which was published in NeurIPS 2024.

## Acknowledgements
Huge thanks to the authors of the following open-source projects:
- [MMSA](https://github.com/thuiar/MMSA)
- [LNLN](https://github.com/Haoyu-ha/LNLN)
- [P-RMF](https://github.com/aoqzhu/P-RMF)

## Training

You can quickly run the code with the following command:

```bash
bash train.sh
