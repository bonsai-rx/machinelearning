using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torchvision;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Vision
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Normalize
    {       
        private ITransform inputTransform;

        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            double[] means = [0.485f, 0.456f, 0.406f];
            double[] stds = [0.229f, 0.224f, 0.225f];

            inputTransform = transforms.Compose(
                transforms.Normalize(means, stds)
            );

            return source.Select(tensor => {
                return inputTransform.call(tensor);
            });
        }
    }
}