using static TorchSharp.torch;
using TorchSharp;
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

[Combinator]
[ResetCombinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Beta : IScalarTypeProvider
{
    [Browsable(false)]
    public ScalarType Type => ScalarType.Float32;

    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Concentration1 { get; set; }

    /// <summary>
    /// The values of concentration 1 in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Concentration1))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string Concentration1Xml
    {
        get => TensorConverter.ConvertToString(Concentration1, Type);
        set => Concentration1 = TensorConverter.ConvertFromString(value, Type);
    }

    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Concentration0 { get; set; }

    /// <summary>
    /// The values of concentration 0 in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Concentration0))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string Concentration0Xml
    {
        get => TensorConverter.ConvertToString(Concentration0, Type);
        set => Concentration0 = TensorConverter.ConvertFromString(value, Type);
    }

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