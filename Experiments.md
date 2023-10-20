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

But I had this package in my environment. So let's look into the CMakeList.txt file to find out what's wrong:

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

This didn't work. Try another way, set the python executable in the shell file:

    cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON -DPYTHON_EXECUTABLE=/opt/conda/bin/python3.8

But this also failed. Let's give a more careful look at the file and the error message, this process is the compilation by cmake, and the prominent config file is CMakeList.txt. Following the shell script, the path is ./submodules/cutlass/CMakeList.txt. Here we found where the error occurred:

    find_package(Python3 3.5 COMPONENTS Interpreter REQUIRED)

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

From CSDN, I learned a liitle about find_package function:

    Usage:
    find_package(<PackageName> [version] [EXACT] [QUIET] [MODULE]
                [REQUIRED] [[COMPONENTS] [components...]]
                [OPTIONAL_COMPONENTS components...]
                [NO_POLICY_SCOPE])

    REQUIRED：denote that this package is essential, if cannot find the construction process reports error and terminates.
    [COMPONENTS] [components…]：Necessary components in all the packages being searched for, if any one can't be found it failes, similar to REQUIRED, will result in termination of cmake.
    OPTIONAL_COMPONENTS components…：will not affect cmake to continue executing if cannot find；
    NO_POLICY_SCOPE：cmake policy，refer to：cmake_policy
    Note that among all these parameters, only PackageName is necessary, the others are all optional.

It's here where the environment variable got changed. So insert such a violent sentence after:
   
    set(Python3_EXECUTABLE /opt/conda/bin/python3.8)

And this small question is finally solved. 
<p align="center">
  <img src="figure/build_cutlass.png">
</p>

Then, run

    python setup.py install

And within expectation, it won't go smoothly:

    building 'torch_int._CUDA' extension
    /usr/local/cuda/bin/nvcc -Itorch_int/kernels/include -I/opt/conda/lib/python3.8/site-packages/torch/include -I/opt/conda/lib/python3.8/site-packages/torch/include/torch/csrc/api/include -I/opt/conda/lib/python3.8/site-packages/torch/include/TH -I/opt/conda/lib/python3.8/site-packages/torch/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.8 -c torch_int/kernels/linear.cu -o build/temp.linux-x86_64-3.8/torch_int/kernels/linear.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -O3 -std=c++14 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -DCUDA_ARCH=800 -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE="_gcc" -DPYBIND11_STDLIB="_libstdcpp" -DPYBIND11_BUILD_ABI="_cxxabi1011" -DTORCH_EXTENSION_NAME=_CUDA -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80
    torch_int/kernels/linear.cu:4:10: fatal error: cutlass/core_io.h: No such file or directory
        4 | #include <cutlass/core_io.h>
          |          ^~~~~~~~~~~~~~~~~~~
    compilation terminated.
    error: command '/usr/local/cuda/bin/nvcc' failed with exit status 1

This seems to be a problem in inclusion of C++ head file. Parts of the core_io.h file is as follows: 

    #include "include/linear.h"
    #include "include/common.h"

    #include <cutlass/core_io.h>
    #include <cutlass/cutlass.h>
    #include <cutlass/half.h>

    #include <cutlass/gemm/device/gemm.h>
    #include <cutlass/numeric_types.h>
    #include <cutlass/util/host_tensor.h>

There should be somewhere to configure the paths of headfiles. In my knowledge the <> characters are for system or standard head files, and "" are for these added by users. The compiler will search in current path, and then in the standard head file path.

However, in IDEs like VS, we can config the additional include directories. Let's look into the setup.py file to see if there are parameters for this purpose.

    setup(
        name='torch_int',
        ext_modules=[
            cpp_extension.CUDAExtension(
                name='torch_int._CUDA',
                sources=[
                    'torch_int/kernels/linear.cu',
                    'torch_int/kernels/bmm.cu',
                    'torch_int/kernels/fused.cu',
                    'torch_int/kernels/bindings.cpp',
                ],
                include_dirs=['torch_int/kernels/include'],
    ...

This corresponds to the error message above. So I add to include_dirs a path 'submodules/cutlass/include/cutlass', where core_id.h and several other headfiles lie in. 

    building 'torch_int._CUDA' extension
    /usr/local/cuda/bin/nvcc -Itorch_int/kernels/include -I/submodules/cutlass/include/cutlass -I/opt/conda/lib/python3.8/site-packages/torch/include 
    ...
    torch_int/kernels/linear.cu:4:10: fatal error: cutlass/core_io.h: No such file or directory
        4 | #include "cutlass/core_io.h"
        |            ^~~~~~~~~~~~~~~~~~~

Obviously, the path has been read successfully, but still it doesn't work. Why does #include "include/linear.h" work but this doesn't? I can find no reason for this.

Well, later I find myself stupid. When the compiler executes #include "include/linear.h" and #include "include/common.h" ,the headfiles are just found in the relative path of current folder, but I've been thinking they are found under torch_int/kernels/include. I forgot the principle that the whole path is that the include path/current path concatenates the relative path, and this elementary question has taken me several hours.

After addressing this one, the next is:

    torch_int/kernels/linear.cu:10:10: fatal error: cutlass/util/host_tensor.h: No such file or directory
    10 | #include <cutlass/util/host_tensor.h>
        |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
    compilation terminated.

I tried to find the headfile manually, but with no result. Luckily I remember Linux provides easy-to-use commands to do such tasks. If cannot find, it means the construction actually failed, and that's real disaster.

Execute:

    find -name host_tensor.h
    ./submodules/cutlass/tools/util/include/cutlass/util/host_tensor.h

After adding this to path, the include problem is solved. And the following:

    /opt/torch-int/submodules/cutlass/include/cute/util/type_traits.hpp(62): error: namespace "std" has no member "conjunction"

    ...

    /opt/torch-int/submodules/cutlass/include/cute/util/type_traits.hpp(201): error: void_t is not a template

    /opt/torch-int/submodules/cutlass/include/cute/numeric/integral_constant.hpp(41): error: "auto" is not allowed here

    ...

    /opt/torch-int/submodules/cutlass/include/cute/numeric/integral_constant.hpp(98): error: the template argument list of the partial specialization includes a nontype argument whose type depends on a template parameter

    /opt/torch-int/submodules/cutlass/include/cute/numeric/integral_constant.hpp(99): error: "auto" is not allowed here

    /opt/torch-int/submodules/cutlass/include/cute/numeric/integral_constant.hpp(99): warning #842-D: constant "n" is not used in or cannot be deduced from the template argument list of class template "cute::is_constant<<error>, const T &>"

    ...

    /opt/torch-int/submodules/cutlass/include/cute/numeric/integral_constant.hpp(338): error: no instance of function template "cute::abs" matches the argument list
                argument types are: (<error-type>)

    ...

    /opt/torch-int/submodules/cutlass/include/cute/numeric/integral_constant.hpp(342): error: no instance of function template "cute::max" matches the argument list
                argument types are: (<error-type>, <error-type>)

    ...

    Error limit reached.
    100 errors detected in the compilation of "torch_int/kernels/linear.cu".
    Compilation terminated.

It's supposed that these are because the C++ version is too low.