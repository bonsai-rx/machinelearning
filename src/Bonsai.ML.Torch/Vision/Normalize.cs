using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace Bonsai.ML.Torch.Vision
{
    [Combinator]
    [Description("Normalizes the input tensor with the mean and standard deviation.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Normalize
    {
        public double[] Means { get; set; } = [ 0.1307 ];
        public double[] StdDevs { get; set; } = [ 0.3081 ];
        private ITransform transform = null;

        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor => {
                transform ??= transforms.Normalize(Means, StdDevs, tensor.dtype, tensor.device);
                return transform.call(tensor);
            });
        }
    }
}