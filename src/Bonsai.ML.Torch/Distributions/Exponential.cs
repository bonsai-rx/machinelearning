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
public class Exponential : IScalarTypeProvider
{
    [XmlIgnore]
    [Browsable(false)]
    public ScalarType Type => ScalarType.Float32;

    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Rate { get; set; }

    /// <summary>
    /// The values of the rates in XML string format.
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

    public IObservable<TorchSharp.Modules.Exponential> Process()
    {
        return Observable.Return(distributions.Exponential(Rate, generator: Generator));
    }

    public IObservable<TorchSharp.Modules.Exponential> Process(IObservable<torch.Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Exponential(Rate, generator: Generator);
        });
    }

    public IObservable<TorchSharp.Modules.Exponential> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Exponential(Rate, generator: Generator));
    }
}