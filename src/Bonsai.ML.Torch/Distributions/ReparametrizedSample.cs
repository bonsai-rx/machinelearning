using Bonsai;
using System;
using System.Reactive.Linq;
using System.ComponentModel;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Distributions;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class ReparametrizedSample
{
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] SampleShape { get; set; }

    public IObservable<Tensor> Process(IObservable<distributions.Distribution> source)
    {
        return source.Select(distribution => distribution.rsample(SampleShape));
    }
}