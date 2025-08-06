using Bonsai;
using System;
using System.Reactive.Linq;
using System.ComponentModel;
using static TorchSharp.torch;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

[Combinator]
[ResetCombinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class LogProbability
{
    [XmlIgnore]
    public Tensor Values { get; set; }

    public IObservable<Tensor> Process(IObservable<distributions.Distribution> source)
    {
        return source.Select(distribution => distribution.log_prob(Values));
    }

    public IObservable<Tensor> Process(IObservable<Tuple<distributions.Distribution, Tensor>> source)
    {
        return source.Select((input) => input.Item1.log_prob(input.Item2));
    }

    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, distributions.Distribution>> source)
    {
        return source.Select((input) => input.Item2.log_prob(input.Item1));
    }
}