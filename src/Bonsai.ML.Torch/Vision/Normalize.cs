using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace Bonsai.ML.Torch.Vision
{
    /// <summary>
    /// Normalizes the input tensor with the mean and standard deviation.
    /// </summary>
    [Combinator]
    [Description("Normalizes the input tensor with the mean and standard deviation.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Normalize
    {
        /// <summary>
        /// The mean values for each channel.
        /// </summary>
        [Description("The mean values for each channel.")]
        public double[] Means { get; set; } = [ 0.1307 ];

        /// <summary>
        /// The standard deviation values for each channel.
        /// </summary>
        [Description("The standard deviation values for each channel.")]
        public double[] StdDevs { get; set; } = [ 0.3081 ];

        private ITransform transform = null;

        /// <summary>
        /// Normalizes the input tensor with the mean and standard deviation.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor => {
                transform ??= transforms.Normalize(Means, StdDevs, tensor.dtype, tensor.device);
                return transform.call(tensor);
            }).Finally(() => transform = null);
        }
    }
}