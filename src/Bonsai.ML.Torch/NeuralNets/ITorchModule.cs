using static TorchSharp.torch;

namespace Bonsai.ML.Torch.NeuralNets
{
    public interface ITorchModule
    {
        public nn.Module Module { get; }
        public Tensor forward(Tensor tensor);
    }
}
