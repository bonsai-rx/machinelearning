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
            inputTransform = transforms.Normalize(new double[] { 0.1307 }, new double[] { 0.3081 });

            return source.Select(tensor => {
                return inputTransform.call(tensor);
            });
        }
    }
}