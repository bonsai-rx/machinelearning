using static TorchSharp.torch;

namespace Bonsai.ML.Torch.NeuralNets
{
    /// <summary>
    /// Represents an interface for a Torch module.
    /// </summary>
    public interface ITorchModule
    {
        /// <summary>
        /// The module.
        /// </summary>
        public nn.Module Module { get; }

        /// <summary>
        /// Runs forward inference on the input tensor using the specified model.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public Tensor Forward(Tensor tensor);
    }
}
