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
public class Binomial : IScalarTypeProvider
{
    [XmlIgnore]
    [Browsable(false)]
    public ScalarType Type => ScalarType.Float32;

    [TypeConverter(typeof(TensorConverter))]
    public Tensor Count { get; set; }

    [TypeConverter(typeof(TensorConverter))]
    public Tensor Probabilities { get; set; }

    [XmlIgnore]
    public torch.Generator Generator { get; set; } = null;

    public IObservable<TorchSharp.Modules.Binomial> Process()
    {
        return Observable.Return(distributions.Binomial(Count, Probabilities, generator: Generator));
    }

    public IObservable<TorchSharp.Modules.Binomial> Process(IObservable<torch.Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Binomial(Count, Probabilities, generator: Generator);
        });
    }

    public IObservable<TorchSharp.Modules.Binomial> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Binomial(Count, Probabilities, generator: Generator));
    }
}