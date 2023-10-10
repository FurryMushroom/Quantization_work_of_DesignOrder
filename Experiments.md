# Olive
## Source code interpretation
The core part of the code lies in olive_quantization/antquant/quant_modules.py. Generally quantization works don't disengage the routine of initializing grids and discretizing values to the grids. In simple circumstances clamp and round functions get used, but Olive uses quant_cuda.quant to complete this. I have no knowledge about the package and feel free to tell if you know anything about that.

The foolowing is the part coping with outlier-victim pairs:

    quant_data = quant_data.view(-1)                
    mask = quant_data.abs() > 32
    victim_odd = torch.roll(mask, 1, -1) # roll the tensor 1 unit on the -1 dim, shift right 1 unit
    victim_odd[::2] = 0
    victim_even = torch.roll(mask & (~victim_odd), -1, -1)
    victim_even[1::2] = 0
    victim = victim_even | victim_odd
    quant_data = quant_data * (~victim)

The authors were writing elegant and glamorous code. But this seems not to correspond with the paper content, which says in a outlier-outlier pair the bigger one will be retained.

Olive has really inspiring experiental results, we tried to reproduce the accuracy results on missions requiring less computational resources. And because hardware doesn't support the FP-based formats, it's predictable there won't be real savings on memory or decrease on inference time. We only conduct the INT4 experiments.

## Results
Reproduced:
| Model | CoLA  | SST-2 | MNLI | QQP | MPRC |
| :----:| :----: | :----: | :----: | :----: | :----: |
| BERT-base | 59.33 | 91.86 | 83.78 | 90.38 | 88.73 |
| BART-base | 53.30 | 93.58 | 85.22 | 91.38 | 86.03 |
| BERT-large | 64.39 | 93.35 | 86.50 | 91.03 | 87.99 |

Original:
| Model | CoLA  | SST-2 | MNLI | QQP | MPRC |
| :----:| :----: | :----: | :----: | :----: | :----: |
| BERT-base | 59.99 | 92.55 | 83.83 | 90.34 | 87.75 |
| BART-base | 52.69 | 94.15 | 85.54 | 91.37 | 85.29 |
| BERT-large | 63.23 | 92.43 | 84.80 | 90.11 | 87.01 |

Reropduced:
| Model | SQuAD v1.1  | SQuAD v2.0 |
| :----:| :----: | :----: |
| BERT-base | 86.31 / 78.06 | 75.90 / 71.93 |
| BART-base | 88.11 / 79.64  | 78.04 / 74.37 |


Oringinal:

| Model | SQuAD v1.1  | SQuAD v2.0 |
| :----:| :----: | :----: |
| BERT-base | 86.38 / 78.24 | 75.67 / 71.54 |
| BART-base | 88.07 / 79.81  | 77.70 / 74.08 |

We can see the reproduced results are consistent with the original. However, I lacked a means to monitor the gpu memory occupation, inference time before and after olive quantization isn't experienmented either.  

# Smoothquant
When referring to the github repo of smoothquant I discovered peers discussing auto GPTQ. That seems an approach devised earlier and also encapsulated, so I'll also conduction investigation and maybe experiment on it.

