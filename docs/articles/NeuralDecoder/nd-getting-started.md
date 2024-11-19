# Getting Started

To get started using the Bonsai.ML.NeuralDecoder package, please read below or get started on the demo in the [Neural Decoding example guide](~/examples/README.md).

## Algorithm

The neural decoder consists of a bayesian state space point process model. With this model, latent variables such as an animals position can be decoded from neural activity. To read more about the theory behind the model and how the algorithm works, we refer the reader to: Denovellis, E.L., Gillespie, A.K., Coulter, M.E., et al. Hippocampal replay of experience at real-world speeds. eLife 10, e64505 (2021). https://doi.org/10.7554/eLife.64505.

## Installation

### Python

To install the python package needed to use the package, run the following:

```
cd \path\to\examples\NeuralDecoding\PositionDecodingFromHippocampus
python -m venv .venv
.\.venv\Scripts\activate
pip install git+https://github.com/ncguilbeault/bayesian-neural-decoder.git
```

You can test whether the installation was successful by launching python and running

```python
import bayesian_neural_decoder
```
