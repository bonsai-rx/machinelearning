using static TorchSharp.torch;
using TorchSharp;
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates an Exponential probability distribution parameterized by rate.
/// Emits a TorchSharp distribution module that can be sampled or queried for probabilities.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates an Exponential distribution with the specified rate parameter.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Exponential : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Exponential"/> class.
    /// </summary>
    public Exponential()
    {
        RegisterTensor(
            () => _rate,
            value => _rate = value);
    }

    private Tensor _rate;
    /// <summary>
    /// Rate parameter (> 0). Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Rate parameter (> 0). Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
    public Tensor Rate
    {
        get => _rate;
        set => _rate = value;
    }

    /// <summary>
    /// The values of the rates in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Rate))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string RateXml
    {
        get => TensorConverter.ConvertToString(_rate, Type);
        set => _rate = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Optional random number generator to use when sampling. If null, TorchSharp's global RNG is used.
    /// </summary>
    [XmlIgnore]
    public Generator Generator { get; set; } = null;

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Exponential"/> distribution using the configured parameters and optional <see cref="Generator"/>.
    /// </summary>
    /// <returns>An observable that emits the constructed Exponential distribution.</returns>
    public IObservable<TorchSharp.Modules.Exponential> Process()
    {
        return Observable.Return(distributions.Exponential(Rate, generator: Generator));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Exponential"/> distribution for each incoming RNG <see cref="torch.Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Exponential distributions.</returns>
    public IObservable<TorchSharp.Modules.Exponential> Process(IObservable<Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Exponential(Rate, generator: Generator);
        });
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Exponential"/> distribution
    /// constructed from the configured parameters and current <see cref="Generator"/>.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Exponential distributions.</returns>
    public IObservable<TorchSharp.Modules.Exponential> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Exponential(Rate, generator: Generator));
    }
}