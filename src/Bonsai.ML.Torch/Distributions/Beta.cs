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
public class Beta : IScalarTypeProvider
{
    [XmlIgnore]
    [Browsable(false)]
    public ScalarType Type => ScalarType.Float32;

    [TypeConverter(typeof(TensorConverter))]
    public Tensor Concentration1 { get; set; }

    [TypeConverter(typeof(TensorConverter))]
    public Tensor Concentration0 { get; set; }

    [XmlIgnore]
    public torch.Generator Generator { get; set; } = null;

    public IObservable<TorchSharp.Modules.Beta> Process()
    {
        return Observable.Return(distributions.Beta(Concentration1, Concentration0, generator: Generator));
    }

    public IObservable<TorchSharp.Modules.Beta> Process(IObservable<torch.Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Beta(Concentration1, Concentration0, generator: Generator);
        });
    }

    public IObservable<TorchSharp.Modules.Beta> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Beta(Concentration1, Concentration0, generator: Generator));
    }
}