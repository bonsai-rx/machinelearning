using static TorchSharp.torch;
using TorchSharp;
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a Binomial probability distribution with a given number of trials and success probability.
/// Emits a TorchSharp distribution module that can be sampled or queried for log-probabilities.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a Binomial distribution with count (number of trials) and probability of success p.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Binomial : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Binomial"/> class.
    /// </summary>
    public Binomial()
    {
        RegisterTensor(
            () => _count,
            value => _count = value);
        RegisterTensor(
            () => _probabilities,
            value => _probabilities = value);
    }

    private Tensor _count;
    /// <summary>
    /// Number of trials (non-negative). Can be a scalar or tensor. If tensor, values should be non-negative integers.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Number of trials (non-negative). Can be a scalar or tensor.")]
    public Tensor Count
    {
        get => _count;
        set => _count = value;
    }

    /// <summary>
    /// The values of count in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Count))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string CountXml
    {
        get => TensorConverter.ConvertToString(Count, Type);
        set => Count = TensorConverter.ConvertFromString(value, Type);
    }

    private Tensor _probabilities;
    /// <summary>
    /// Probability of success p in [0, 1]. Can be a scalar or tensor; the shape should be broadcastable to <see cref="Count"/>.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Probability of success p in [0, 1]. Can be a scalar or tensor; shape should broadcast with Count.")]
    public Tensor Probabilities
    {
        get => _probabilities;
        set => _probabilities = value;
    }

    /// <summary>
    /// The values of probabilities in XML string format.
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
    /// Creates a <see cref="TorchSharp.Modules.Binomial"/> distribution using the configured parameters and optional <see cref="Generator"/>.
    /// </summary>
    /// <returns>An observable that emits the constructed Binomial distribution.</returns>
    public IObservable<TorchSharp.Modules.Binomial> Process()
    {
        return Observable.Return(distributions.Binomial(Count, Probabilities, generator: Generator));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Binomial"/> distribution for each incoming RNG <see cref="torch.Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Binomial distributions.</returns>
    public IObservable<TorchSharp.Modules.Binomial> Process(IObservable<Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Binomial(Count, Probabilities, generator: Generator);
        });
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Binomial"/> distribution
    /// constructed from the configured parameters and current <see cref="Generator"/>.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Binomial distributions.</returns>
    public IObservable<TorchSharp.Modules.Binomial> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Binomial(Count, Probabilities, generator: Generator));
    }
}