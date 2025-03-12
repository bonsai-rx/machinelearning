# Bonsai.ML.Torch - Overview

The Torch package provides a Bonsai interface to interact with the [TorchSharp](https://github.com/dotnet/TorchSharp) package, a C# implementation of the torch library. This package adds powerful functionality into Bonsai, namely to perform tensor manipulations, type conversions, complex linear algebra, deep neural networks, and support for GPU processing.

## Installation Guide

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

Finally, extract the zip folder and copy the contents of the `lib` folder into the `Extensions` folder of your bonsai installation directory.

> [!WARNING]
> It is advised to install only one of the above CPU or GPU packages, not both. The GPU package contains all of the support of the CPU package. 

## Getting Started

The aim of the `Bonsai.ML.Torch` package is to integrate the [TorchSharp](https://github.com/dotnet/TorchSharp) package, a C# wrapper around the powerful torch library, into Bonsai. In the current version, the package primarily provides tooling and functionality for users to interact with and manipulate `Tensor` objects, the core data type of libtorch which underlies many of the advanced torch operations. Additionally, the package provides some capabilities for defining neural network architectures, running forward inference, and learning via back propagation.

## Tensor Operations
The package provides several ways to work with tensors. Users can initialize tensors, (`Ones`, `Zeros`, etc.), create tensors from .NET data types, (`ToTensor`), and define custom tensors using Python-like syntax (`CreateTensor`). Tensors can be converted back to .NET types using the `ToArray` node (for flattening tensors into a unidimensional array) or the `ToNDArray` node (for preserving multidimensional array shapes). Furthermore, the `Tensor` data types contains many extension methods which can be used via scripting, such as using `ExpressionTransform` (for example, it.sum() to sum a tensor, or it.T to transpose), and works with overloaded operators, for example, `Zip` -> `Multiply`. It is also possible to use the `ExpressionTransform` node to access individual elements of a tensor, using the syntax `it.ReadCpuT(0)` where `T` is a primitive .NET data type (i.e. `Single`, `Double`, `Int64`, etc.).


## Running on the GPU
Users must be explicit about running tensors on the GPU. First, the `InitializeDeviceType` node must run with a CUDA-compatible GPU. Afterwards, tensors are moved to the GPU using the `ToDevice` node. Converting tensors back to .NET data types requires moving the tensor back to the CPU before converting.

## Neural Networks
The package provides initial support for working with torch `Module` objects, the core type representing deep neural networks. The `LoadModuleFromArchitecture` node allows users to select from a list of common architectures, and can optionally load in pretrained weights from disk. Additionally, the package supports loading `TorchScript` modules with the `LoadScriptModule` node, which enables users to use torch modules saved in the `.pt` file format. Users can then use the `Forward` node to run inference and the `Backward` node to run back propagation.   