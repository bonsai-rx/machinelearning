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
public class Geometric : IScalarTypeProvider
{
    [XmlIgnore]
    [Browsable(false)]
    public ScalarType Type => ScalarType.Float32;

    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Probabilities { get; set; }

    /// <summary>
    /// The values of the probabilities in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Probabilities))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ProbabilitiesXml
    {
        get => TensorConverter.ConvertToString(Probabilities, Type);
        set => Probabilities = TensorConverter.ConvertFromString(value, Type);
    }

    [XmlIgnore]
    public torch.Generator Generator { get; set; } = null;

    public IObservable<TorchSharp.Modules.Geometric> Process()
    {
        return Observable.Return(distributions.Geometric(Probabilities, generator: Generator));
    }

    public IObservable<TorchSharp.Modules.Geometric> Process(IObservable<torch.Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Geometric(Probabilities, generator: Generator);
        });
    }

    public IObservable<TorchSharp.Modules.Geometric> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Geometric(Probabilities, generator: Generator));
    }
}