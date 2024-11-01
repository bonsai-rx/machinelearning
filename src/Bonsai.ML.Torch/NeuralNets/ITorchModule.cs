using static TorchSharp.torch;

namespace Bonsai.ML.Torch.NeuralNets
{
    public interface ITorchModule
    {
        public Tensor forward(Tensor tensor);
    }
}
