# Bonsai.ML.Torch Overview

The Torch package provides a Bonsai interface to interact with the [TorchSharp](https://github.com/dotnet/TorchSharp) package, a C# implementation of the torch library.

## General Guide

The Bonsai.ML.Torch package can be installed through the Bonsai Package Manager and depends on the TorchSharp library. Additionally, running the package requires installing the specific torch DLLs needed for your desired application. The steps for installing are outlined below.

### Running on the CPU 
For running the package using the CPU, the `TorchSharp-cpu` library can be installed though the `nuget` package source.

### Running on the GPU
To run torch on the GPU, you first need to ensure that you have a CUDA compatible device installed on your system. 

Next, you must follow the [CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) or the [guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Make sure to install the correct `CUDA v12.1` version [found here](https://developer.nvidia.com/cuda-12-1-0-download-archive). Ensure that you have the correct CUDA version (v12.1) installed, as `TorchSharp` currently only supports this version.

Next, you need to install the `cuDNN v9` library following the [guide for Windows](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/windows.html) or the [guide for Linux](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html). Again, you need to ensure you have the correct version installed (v9). You should consult [nvidia's support matrix](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html) to ensure the versions of CUDA and cuDNN you installed are compatible with your specific OS, graphics driver, and hardware.

Once complete, you need to install the cuda-compatible torch libraries and place them into the correct location. You can download the libraries from [the pytorch website](https://pytorch.org/get-started/locally/) with the following options selected:

- PyTorch Build: Stable (2.5.1)
- OS: [Your OS]
- Package: LibTorch
- Language: C++/Java
- Compute Platform: CUDA 12.1

Finally, extract the zip folder and copy all of the DLLs into the `Extensions` folder of your bonsai installation directory.