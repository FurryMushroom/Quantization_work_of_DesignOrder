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

# Smoothquant
When referring to the github repo of smoothquant I discovered peers discussing auto GPTQ. That seems an approach devised earlier and also encapsulated, so I'll also conduction investigation and maybe experiment on it.

Smoothquant relies on torch-int, both are repos created and maintained by mit-han-lab. Note that there are two jupyter notebook files to reproduce the experiment results, but one is already unusable.

See issue #33 and issue #58, of which the latter was created by me.

The configuration and compilation of torch-int seem to be what's really challenging for me student with developing knowledge and capabilities.

Following the instructions in the readme file, error just occured when executing:

    git clone --recurse-submodules https://github.com/Guangxuan-Xiao/torch-int.git

The error information is as follows:

    Submodule 'submodules/cutlass' (git@github.com:NVIDIA/cutlass.git) registered for path 'submodules/cutlass'
    Cloning into '/opt/conda/bin/torch-int/submodules/cutlass'...
    kex_exchange_identification: Connection closed by remote host
    fatal: Could not read from remote repository.

I felt strange that the Nv repo, which should be in well maintenance cannot be found. I created the issue #22 but later I found the answer to the question in issue #3.

The ssh repo address sometimes fails to function, use https instead.

    modify the address in .gitmodules as: https://github.com/NVIDIA/cutlass.git 

This is familiar to any github usr. I created a new repo and pulled the content of torch-int repo, having the address fixed, and pushed it to Github. Then I was pulling this repo in the docker environment of the company, but somethings were missed.

    make: *** No targets specified and no makefile found.  Stop.

The cloning of cutlass repo wasn't successful yet. I quickly found it was the folder didn't exist. I suspected it's because git ignores empty folders, and results given by search engine confirmed this.
I tried ways of creating a .gitignore or .gitkeep file, however they resulted in a non-empty folder thus obstructing the cloning.

Why don't I just download zip, uncompress and upload? This seems foolish, but it worked. 

The next problem is a version conflict.

    CMake Error at CMakeLists.txt:29 (cmake_minimum_required):
    CMake 3.19 or higher is required.  You are running version 3.16.3

I used pip to install cmake of a higher version, and it was within expectaion that the system still used the original. Following tutorials of CSDN, somewhat like Chinese version of stackoverflow:

Deleted cmake of the old version:

    sudo rm -rf /usr/bin/cmake

The docker env didn't even have sudo. Run:

    apt-get install sudo

Then, searched for our new cmake executable file:

    which cmake

There was no response. But thanks for pip, which told me where the package was:

    Requirement already satisfied: cmake==3.20.2 in /opt/conda/lib/python3.8/site-packages (3.20.2)

So I built the soft link by:

    sudo ln -s /opt/conda/lib/python3.8/site-packages/cmake/data/bin/cmake /usr/bin/cmake

And it worked. 

New problem stroke in the configuration process by cmake:

    -- CMake Version: 3.20.2
    -- The CXX compiler identification is GNU 9.3.0
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Check for working CXX compiler: /usr/bin/g++ - skipped
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- The CUDA compiler identification is NVIDIA 11.6.55
    -- Detecting CUDA compiler ABI info
    -- Detecting CUDA compiler ABI info - done
    -- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
    -- Detecting CUDA compile features
    -- Detecting CUDA compile features - done
    -- CUDART: /usr/local/cuda/lib64/libcudart.so
    -- CUDA Driver: /usr/local/cuda/lib64/stubs/libcuda.so
    -- NVRTC: /usr/local/cuda/lib64/libnvrtc.so
    -- Default Install Location: install
    -- Found Python3: /usr/bin/python3.8 (found suitable version "3.8.10", minimum required is "3.5") found components: Interpreter 
    CMake Error at CMakeLists.txt:100 (message):
    Error installing cutlass_library package.  See
    /opt/torch-int/submodules/cutlass/build/cutlass_library_installation.log


    -- Configuring incomplete, errors occurred!
    See also "/opt/torch-int/submodules/cutlass/build/CMakeFiles/CMakeOutput.log".
    make: *** No targets specified and no makefile found.  Stop.

And what the file contained seemed really easy to solve:

        Traceback (most recent call last):
    File "/opt/torch-int/submodules/cutlass/python/setup_library.py", line 33, in <module>
        from setuptools import setup
    ModuleNotFoundError: No module named 'setuptools'

But I had this package in my environment. So let's look into the cmakelist.txt file to find out what's wrong:

    # Install cutlass_library Python package
    execute_process(
    WORKING_DIRECTORY ${CUTLASS_DIR}/python
    COMMAND ${Python3_EXECUTABLE} ${CUTLASS_DIR}/python/setup_library.py develop --user
    RESULT_VARIABLE cutlass_lib_GENERATOR_INSTALL_RESULT
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/cutlass_library_installation.log
    ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/cutlass_library_installation.log
    )

    if(NOT cutlass_lib_GENERATOR_INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "Error installing cutlass_library package. See ${CMAKE_CURRENT_BINARY_DIR}/cutlass_library_installation.log")
    endif()

It's using Python3_EXECUTABLE as the python executable file, and that is /usr/bin/python3.8. However I have always been using python or python3, that may be what the problem lies in.

Lets do some experiments:

    root@439f6277dda5:/opt/torch-int# /opt/conda/bin/python3.8
    Python 3.8.12 | packaged by conda-forge | (default, Oct 12 2021, 21:59:51) 
    [GCC 9.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import setuptools
    >>> import ssss
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    ModuleNotFoundError: No module named 'ssss'

Well, it's not the problem, but the default python executable path is set usr/bin/python3.8. Let me figure out how to modify. 

Add this to the file build_cutlass.sh:

export Python3_EXECUTABLE=/opt/conda/bin/python3.8