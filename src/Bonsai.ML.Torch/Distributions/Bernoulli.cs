using static TorchSharp.torch;
using TorchSharp;
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Bernoulli : IScalarTypeProvider
{
    [XmlIgnore]
    [Browsable(false)]
    public ScalarType Type => ScalarType.Float32;

    [TypeConverter(typeof(TensorConverter))]
    public Tensor Probability { get; set; }

    [XmlIgnore]
    public torch.Generator Generator { get; set; } = null;

    public IObservable<TorchSharp.Modules.Bernoulli> Process()
    {
        return Observable.Return(distributions.Bernoulli(Probability, generator: Generator));
    }

    public IObservable<TorchSharp.Modules.Bernoulli> Process(IObservable<torch.Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Bernoulli(Probability, generator: Generator);
        });
    }

    public IObservable<TorchSharp.Modules.Bernoulli> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Bernoulli(Probability, generator: Generator));
    }
}