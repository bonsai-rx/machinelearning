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
public class Gamma : IScalarTypeProvider
{
    [XmlIgnore]
    [Browsable(false)]
    public ScalarType Type => ScalarType.Float32;

    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Concentration { get; set; }

    /// <summary>
    /// The values of the concentration in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Concentration))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ConcentrationXml
    {
        get => TensorConverter.ConvertToString(Concentration, Type);
        set => Concentration = TensorConverter.ConvertFromString(value, Type);
    }

    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Rate { get; set; }

    /// <summary>
    /// The values of the rate in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Rate))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string RateXml
    {
        get => TensorConverter.ConvertToString(Rate, Type);
        set => Rate = TensorConverter.ConvertFromString(value, Type);
    }

    [XmlIgnore]
    public torch.Generator Generator { get; set; } = null;

    public IObservable<TorchSharp.Modules.Gamma> Process()
    {
        return Observable.Return(distributions.Gamma(Concentration, Rate, generator: Generator));
    }

    public IObservable<TorchSharp.Modules.Gamma> Process(IObservable<torch.Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Gamma(Concentration, Rate, generator: Generator);
        });
    }

    public IObservable<TorchSharp.Modules.Gamma> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Gamma(Concentration, Rate, generator: Generator));
    }
}