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
public class Cauchy : IScalarTypeProvider
{
    [Browsable(false)]
    public ScalarType Type => ScalarType.Float32;

    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Locations { get; set; }

    /// <summary>
    /// The values of the locations in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Locations))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string LocationsXml
    {
        get => TensorConverter.ConvertToString(Locations, Type);
        set => Locations = TensorConverter.ConvertFromString(value, Type);
    }

    [TypeConverter(typeof(TensorConverter))]
    public Tensor Scales { get; set; }

    /// <summary>
    /// The values of the scales in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Scales))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ScalesXml
    {
        get => TensorConverter.ConvertToString(Scales, Type);
        set => Scales = TensorConverter.ConvertFromString(value, Type);
    }

    [XmlIgnore]
    public torch.Generator Generator { get; set; } = null;

    public IObservable<TorchSharp.Modules.Cauchy> Process()
    {
        return Observable.Return(distributions.Cauchy(Locations, Scales, generator: Generator));
    }

    public IObservable<TorchSharp.Modules.Cauchy> Process(IObservable<torch.Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Cauchy(Locations, Scales, generator: Generator);
        });
    }

    public IObservable<TorchSharp.Modules.Cauchy> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Cauchy(Locations, Scales, generator: Generator));
    }
}