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
public class Dirichlet : IScalarTypeProvider
{
    [XmlIgnore]
    [Browsable(false)]
    public ScalarType Type => ScalarType.Float32;

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
    public torch.Generator Generator { get; set; } = null;

    public IObservable<TorchSharp.Modules.Dirichlet> Process()
    {
        return Observable.Return(distributions.Dirichlet(Concentration, generator: Generator));
    }

    public IObservable<TorchSharp.Modules.Dirichlet> Process(IObservable<torch.Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Dirichlet(Concentration, generator: Generator);
        });
    }

    public IObservable<TorchSharp.Modules.Dirichlet> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Dirichlet(Concentration, generator: Generator));
    }
}