# Bonsai.ML.HiddenMarkovModels Overview

The HiddenMarkovModels package provides a Bonsai interface to interact with the [ssm](https://github.com/lindermanlab/ssm) package.

## General Guide

Since the package relies on both Bonsai and Python, installation steps for both are required. Detailed instructions are provided for installing the package in a new environment, integrating it with existing workflows, and running examples from the example folder.

- To install the package for integrating with existing workflows, see the [Installation Guide](#installation-guide).
- To get started with integrating the package into workflows, see the [Getting Started](hmm-getting-started.md) section.
- To test the specific examples provided, check out the [Examples](~/examples/README.md) tab.

## Installation Guide

### Dependencies

To get started, you must install the following tools:

- [Python (v3.10)](https://www.python.org/downloads/) 
- [dotnet-sdk (v8)](https://dotnet.microsoft.com/en-us/download)
- [Git](https://git-scm.com/downloads)
- [Bonsai-Rx Templates tool](https://www.nuget.org/packages/Bonsai.Templates)

  > [!TIP]
  > Install Python through the standard installer and add to the system PATH.

### Installation Guide - Windows

#### Creating New Project Environment

1. Open the terminal and create a project folder:
    ```cmd
    cd ~\Desktop
    mkdir HiddenMarkovModels
    cd .\HiddenMarkovModels
    ```

2. Create a Python virtual environment:
    ```cmd
    python -m venv .venv
    ```

3. Create a Bonsai environment:
    ```cmd
    dotnet new bonsaienv
    ```

#### Python Environment Setup

1. Activate the Python environment:
    ```cmd
    .\.venv\Scripts\activate
    ```

2. Install the ssm package:
    ```cmd
    pip install numpy cython
    pip install ssm@git+https://github.com/lindermanlab/ssm@6c856ad3967941d176eb348bcd490cfaaa08ba60
    ```

3. Verify installation:
    ```python
    import ssm
    ```

#### Bonsai Environment Setup

1. Launch Bonsai:
    ```cmd
    .bonsai\Bonsai.exe
    ```

2. Install the `Bonsai.ML.HiddenMarkovModels` package from the Package Manager.
    > [!TIP]
    > You can quickly search for the package by entering `Bonsai.ML.HiddenMarkovModels` into the search bar.

### Installation Guide - Linux

#### Creating New Project Environment

1. Create a project folder:
    ```cmd
    cd ~/Desktop
    mkdir HiddenMarkovModels
    cd HiddenMarkovModels
    ```

2. Create a Python virtual environment:
    ```cmd
    python3 -m venv .venv
    ```
    > [!TIP]
    > Install the virtual environment package if needed:
    > ```cmd
    > sudo apt install python3.10-venv
    > ```

3. Create a Bonsai environment:
    ```cmd
    dotnet new bonsaienv
    ```
    > [!NOTE]
    > This step uses the [Bonsai Linux Environment Template tool](https://github.com/ncguilbeault/bonsai-linux-environment-template) for easy creation of bonsai environments on Linux.
    > See [this discussion](https://github.com/orgs/bonsai-rx/discussions/1101) for more information on getting Bonsai running on Linux.

#### Python Environment Setup

1. Activate the Python environment:
    ```cmd
    source .venv/bin/activate
    ```

2. Install the ssm package:
    ```cmd
    pip install numpy cython
    pip install ssm@git+https://github.com/lindermanlab/ssm@6c856ad3967941d176eb348bcd490cfaaa08ba60
    ```

3. Verify installation:
    ```python
    import ssm
    ```

#### Bonsai Environment Setup

1. Activate and launch Bonsai:
    ```cmd
    source .bonsai/activate
    bonsai
    ```

2. Install the `Bonsai.ML.HiddenMarkovModels` package from the Package Manager.
    > [!TIP]
    > You can quickly search for the package by entering `Bonsai.ML.HiddenMarkovModels` into the search bar.
