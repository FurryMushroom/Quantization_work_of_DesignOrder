# Olive
## Source code interpretation
The source code of the paper is at:

    https://github.com/clevercool/ANT-Quantization

The core part of the code lies in olive_quantization/antquant/quant_modules.py. Generally quantization works don't disengage the routine of initializing grids and discretizing values to the grids. In simple circumstances clamp and round functions get used, but Olive uses quant_cuda.quant to complete this. I have no knowledge about the package and feel free to tell if you know anything about that.

The following is the part coping with outlier-victim pairs:

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
## The process
There was no obstacles worthy to be noted, the source code has a commendable quality. Follow the instructions of the README file, the executable shell file for results on relatively small models is in olive_quantization/bert/scripts.
You may only need to modify a little here according to your environment and config.
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

Update of 10.11:

I've got several means to monitor GPU memory occupation constantly. First, I was taught that it's not advisable to run front-end training or inference. If the front-end connection breaks or the terminal shut down, 
The process will be killed and this can result in great loss of time and computaional resources. Run command like such:

    nohup [your command] &

This will generate a nohup.out file in current path which records the output you should have seen in the terminal.
But if you want to terminate the process, you need to find its PID and kill accordingly.

    ps -aux | grep "myShellScript.sh" 

or use 

    ps -def | grep "myShellScript.sh"

to search for the PID. And kill the process by

    kill -9 PID

And also there are several ways to continuously monitor the GPU memory:

Flush the nvidia-smi command every 10 seconds:
    nvidia-smi -l 10
The above method displays current and history information at the same time. And this way only shows current info, flushing every 1 second:

    watch -n 1 nvidia-smi

nvtop: 

    sudo apt install nvtop
nvitop:
this can be installed by pip as a python package. Run this command to display overall GPU info:

    nvitop -m full

