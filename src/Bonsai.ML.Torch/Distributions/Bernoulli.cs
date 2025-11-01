using static TorchSharp.torch;
using TorchSharp;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a Bernoulli probability distribution parameterized by event probabilities.
/// Emits a TorchSharp distribution module that can be sampled or queried for log-probabilities.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a Bernoulli distribution with event probabilities and emits a TorchSharp distribution module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Bernoulli : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Bernoulli"/> class.
    /// </summary>
    public Bernoulli()
    {
        RegisterTensor(
            () => _probabilities,
            value => _probabilities = value);
    }

    private Tensor _probabilities;
    /// <summary>
    /// Event probabilities p in [0, 1]. Can be a scalar or a tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Event probabilities p in [0, 1]. Can be a scalar or a tensor; shape sets the batch/event shape of the distribution.")]
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
    /// Creates a <see cref="TorchSharp.Modules.Bernoulli"/> distribution using the configured <see cref="Probabilities"/> and optional <see cref="Generator"/>.
    /// </summary>
    /// <returns>An observable that emits the constructed Bernoulli distribution.</returns>
    public IObservable<TorchSharp.Modules.Bernoulli> Process()
    {
        return Observable.Return(distributions.Bernoulli(Probabilities, generator: Generator));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Bernoulli"/> distribution for each incoming RNG <see cref="torch.Generator"/>,
    /// updating <see cref="Generator"/> and passing it to TorchSharp.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Bernoulli distributions.</returns>
    public IObservable<TorchSharp.Modules.Bernoulli> Process(IObservable<Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Bernoulli(Probabilities, generator: Generator);
        });
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Bernoulli"/> distribution
    /// constructed from the configured <see cref="Probabilities"/> and current <see cref="Generator"/>.
    /// The source values are ignored and used only for timing.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Bernoulli distributions.</returns>
    public IObservable<TorchSharp.Modules.Bernoulli> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Bernoulli(Probabilities, generator: Generator));
    }
}