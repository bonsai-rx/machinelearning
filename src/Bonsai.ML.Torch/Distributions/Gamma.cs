using static TorchSharp.torch;
using TorchSharp;
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a Gamma probability distribution parameterized by concentration and rate.
/// Emits a TorchSharp distribution module that can be sampled or queried for probabilities.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a Gamma distribution with concentration and rate parameters.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Gamma : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Gamma"/> class.
    /// </summary>
    public Gamma()
    {
        RegisterTensor(
            () => _concentration,
            value => _concentration = value);

        RegisterTensor(
            () => _rate,
            value => _rate = value);
    }

    private Tensor _concentration;
    /// <summary>
    /// Concentration parameter (> 0). Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Concentration parameter (> 0). Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
    public Tensor Concentration
    {
        get => _concentration;
        set => _concentration = value;
    }

    /// <summary>
    /// The values of the concentration in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Concentration))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ConcentrationXml
    {
        get => TensorConverter.ConvertToString(_concentration, Type);
        set => _concentration = TensorConverter.ConvertFromString(value, Type);
    }

    private Tensor _rate;
    /// <summary>
    /// Rate parameter (> 0). Can be a scalar or tensor; must be broadcastable with <see cref="Concentration"/>.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Rate parameter (> 0). Can be a scalar or tensor; must broadcast with Concentration.")]
    public Tensor Rate
    {
        get => _rate;
        set => _rate = value;
    }

    /// <summary>
    /// The values of the rate in XML string format.
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
    /// Creates a <see cref="TorchSharp.Modules.Gamma"/> distribution using the configured parameters and optional <see cref="Generator"/>.
    /// </summary>
    /// <returns>An observable that emits the constructed Gamma distribution.</returns>
    public IObservable<TorchSharp.Modules.Gamma> Process()
    {
        return Observable.Return(distributions.Gamma(Concentration, Rate, generator: Generator));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Gamma"/> distribution for each incoming RNG <see cref="torch.Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Gamma distributions.</returns>
    public IObservable<TorchSharp.Modules.Gamma> Process(IObservable<Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Gamma(Concentration, Rate, generator: Generator);
        });
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Gamma"/> distribution
    /// constructed from the configured parameters and current <see cref="Generator"/>.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Gamma distributions.</returns>
    public IObservable<TorchSharp.Modules.Gamma> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Gamma(Concentration, Rate, generator: Generator));
    }
}