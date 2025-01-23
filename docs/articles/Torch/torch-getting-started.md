# Getting Started

The aim of the `Bonsai.ML.Torch` package is to integrate the [TorchSharp](https://github.com/dotnet/TorchSharp) package, a C# wrapper around the powerful libtorch library, into Bonsai. In the current version, the package primarily provides tooling and functionality for users to interact with and manipulate `Tensor`s, the core data type of libtorch which underlies many of the advanced torch operations. Additionally, the package provides some capabilities for defining neural network architectures, running forward inference, and learning via back propogation.

## Tensor Operations
The package provides several ways to work with tensors. Users can initialize tensors, (`Ones`, `Zeros`, etc.), create tensors from .NET data types, (`ToTensor`), and define custom tensors using Python-like syntax (`CreateTensor`). Tensors can be converted back to .NET types using the `ToArray` node (for flattening tensors into a single array) or the `ToNDArray` node (for preserving multidimensional array shapes). Furthermore, the `Tensor` data types contains many extension methods which can be used via scripting, such as using `ExpressionTransform` (for example, it.sum() to sum a tensor, or it.T to transpose), and works with overloaded operators, for example, `Zip` -> `Multiply`. Thus, `ExpressionTransform` can also be used to access individual elements of a tensor, using the syntax `it.data<T>.ReadCpuT(0)` where `T` is a primitive .NET data type.


## Running on the GPU
Users must be explicit about running tensors on the GPU. First, the `InitializeDeviceType` node must run with a CUDA-compatible GPU. Afterwards, tensors are moved to the GPU using the `ToDevice` node. Converting tensors back to .NET data types requires moving the tensor back to the CPU before converting.

## Neural Networks
The package provides initial support for working with torch `Module`s, the conventional object for deep neural networks. The `LoadModuleFromArchitecture` node allows users to select from a list of common architectures, and can optionally load in pretrained weights from disk. Additionally, the package supports loading `TorchScript` modules with the `LoadScriptModule` node, which enables users to use torch modules saved in the `.pt` file format. Users can then use the `Forward` node to run inference and the `Backward` node to run back propogation.   