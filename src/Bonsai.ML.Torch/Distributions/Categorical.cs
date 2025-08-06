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
public class Categorical : IScalarTypeProvider
{
    [Browsable(false)]
    public ScalarType Type => ScalarType.Float32;

    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Probabilities { get; set; }

    /// <summary>
    /// The values of probabilities in XML string format.
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

    public IObservable<TorchSharp.Modules.Categorical> Process()
    {
        return Observable.Return(distributions.Categorical(Probabilities, generator: Generator));
    }

    public IObservable<TorchSharp.Modules.Categorical> Process(IObservable<torch.Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Categorical(Probabilities, generator: Generator);
        });
    }

    public IObservable<TorchSharp.Modules.Categorical> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Categorical(Probabilities, generator: Generator));
    }
}