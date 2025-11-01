using static TorchSharp.torch;
using TorchSharp;
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a Geometric probability distribution parameterized by success probability.
/// Emits a TorchSharp distribution module that can be sampled or queried for probabilities.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a Geometric distribution with the specified success probability.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Geometric : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Geometric"/> class.
    /// </summary>
    public Geometric()
    {
        RegisterTensor(
            () => _probabilities,
            value => _probabilities = value);
    }

    private Tensor _probabilities;
    /// <summary>
    /// Success probability p in [0, 1]. Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Success probability p in [0, 1]. Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
    public Tensor Probabilities
    {
        get => _probabilities;
        set => _probabilities = value;
    }

    /// <summary>
    /// The values of the probabilities in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Probabilities))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ProbabilitiesXml
    {
        get => TensorConverter.ConvertToString(_probabilities, Type);
        set => _probabilities = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Optional random number generator to use when sampling. If null, TorchSharp's global RNG is used.
    /// </summary>
    [XmlIgnore]
    public Generator Generator { get; set; } = null;

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Geometric"/> distribution using the configured parameters and optional <see cref="Generator"/>.
    /// </summary>
    /// <returns>An observable that emits the constructed Geometric distribution.</returns>
    public IObservable<TorchSharp.Modules.Geometric> Process()
    {
        return Observable.Return(distributions.Geometric(Probabilities, generator: Generator));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Geometric"/> distribution for each incoming RNG <see cref="torch.Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Geometric distributions.</returns>
    public IObservable<TorchSharp.Modules.Geometric> Process(IObservable<Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Geometric(Probabilities, generator: Generator);
        });
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Geometric"/> distribution
    /// constructed from the configured parameters and current <see cref="Generator"/>.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Geometric distributions.</returns>
    public IObservable<TorchSharp.Modules.Geometric> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Geometric(Probabilities, generator: Generator));
    }
}