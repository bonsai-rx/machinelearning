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
public class Poisson : IScalarTypeProvider
{
    [XmlIgnore]
    [Browsable(false)]
    public ScalarType Type => ScalarType.Float32;

    [TypeConverter(typeof(TensorConverter))]
    public Tensor Rate { get; set; }

    [XmlIgnore]
    public torch.Generator Generator { get; set; } = null;

    public IObservable<TorchSharp.Modules.Poisson> Process()
    {
        return Observable.Return(distributions.Poisson(Rate, generator: Generator));
    }

    public IObservable<TorchSharp.Modules.Poisson> Process(IObservable<torch.Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Poisson(Rate, generator: Generator);
        });
    }

    public IObservable<TorchSharp.Modules.Poisson> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Poisson(Rate, generator: Generator));
    }
}