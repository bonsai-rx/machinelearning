using static TorchSharp.torch;
using TorchSharp;
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a Cauchy (Lorentz) distribution parameterized by location and scale.
/// Emits a TorchSharp distribution module that can be sampled or queried for probabilities.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a Cauchy distribution with the specified location and scale parameters.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Cauchy : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Cauchy"/> class.
    /// </summary>
    public Cauchy()
    {
        RegisterTensor(
            () => _locations,
            value => _locations = value);

        RegisterTensor(
            () => _scales,
            value => _scales = value);
    }

    private Tensor _locations;
    /// <summary>
    /// Location parameter. Can be a scalar or tensor; shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Location parameter. Can be a scalar or tensor; supports batching.")]
    public Tensor Locations
    {
        get => _locations;
        set => _locations = value;
    }

    /// <summary>
    /// The values of the locations in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Locations))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string LocationsXml
    {
        get => TensorConverter.ConvertToString(_locations, Type);
        set => _locations = TensorConverter.ConvertFromString(value, Type);
    }

    private Tensor _scales;
    /// <summary>
    /// Scale parameter (> 0). Can be a scalar or tensor; must be broadcastable with <see cref="Locations"/>.
    /// </summary>
    [TypeConverter(typeof(TensorConverter))]
    [Description("Scale parameter (> 0). Can be a scalar or tensor; must broadcast with Locations.")]
    public Tensor Scales
    {
        get => _scales;
        set => _scales = value;
    }

    /// <summary>
    /// The values of the scales in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Scales))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ScalesXml
    {
        get => TensorConverter.ConvertToString(_scales, Type);
        set => _scales = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Optional random number generator to use when sampling. If null, TorchSharp's global RNG is used.
    /// </summary>
    [XmlIgnore]
    public Generator Generator { get; set; } = null;

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Cauchy"/> distribution using the configured parameters.
    /// </summary>
    /// <returns>An observable that emits the constructed Cauchy distribution.</returns>
    public IObservable<TorchSharp.Modules.Cauchy> Process()
    {
        return Observable.Return(distributions.Cauchy(Locations, Scales, generator: Generator));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Cauchy"/> distribution for each incoming RNG <see cref="torch.Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Cauchy distributions.</returns>
    public IObservable<TorchSharp.Modules.Cauchy> Process(IObservable<Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Cauchy(Locations, Scales, generator: Generator);
        });
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Cauchy"/> distribution
    /// constructed from the configured parameters and current <see cref="Generator"/>.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Cauchy distributions.</returns>
    public IObservable<TorchSharp.Modules.Cauchy> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Cauchy(Locations, Scales, generator: Generator));
    }
}