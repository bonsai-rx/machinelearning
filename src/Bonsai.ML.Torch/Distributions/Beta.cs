using static TorchSharp.torch;
using TorchSharp;
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a Beta probability distribution parameterized by two concentration parameters (alpha, beta).
/// Emits a TorchSharp distribution module that can be sampled or queried for probabilities.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a Beta distribution with concentration parameters (alpha, beta).")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Beta : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Beta"/> class.
    /// </summary>
    public Beta()
    {
        RegisterTensor(
            () => _concentration1,
            value => _concentration1 = value);

        RegisterTensor(
            () => _concentration0,
            value => _concentration0 = value);
    }

    private Tensor _concentration1;
    /// <summary>
    /// Concentration parameter alpha (> 0). Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Concentration alpha (> 0). Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
    public Tensor Concentration1
    {
        get => _concentration1;
        set => _concentration1 = value;
    }

    /// <summary>
    /// The values of concentration 1 in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Concentration1))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string Concentration1Xml
    {
        get => TensorConverter.ConvertToString(_concentration1, Type);
        set => _concentration1 = TensorConverter.ConvertFromString(value, Type);
    }

    private Tensor _concentration0;
    /// <summary>
    /// Concentration parameter beta (> 0). Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Concentration beta (> 0). Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
    public Tensor Concentration0
    {
        get => _concentration0;
        set => _concentration0 = value;
    }

    /// <summary>
    /// The values of concentration 0 in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Concentration0))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string Concentration0Xml
    {
        get => TensorConverter.ConvertToString(_concentration0, Type);
        set => _concentration0 = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Optional random number generator to use when sampling. If null, TorchSharp's global RNG is used.
    /// </summary>
    [XmlIgnore]
    public Generator Generator { get; set; } = null;

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Beta"/> distribution using the configured parameters and optional <see cref="Generator"/>.
    /// </summary>
    /// <returns>An observable that emits the constructed Beta distribution.</returns>
    public IObservable<TorchSharp.Modules.Beta> Process()
    {
        return Observable.Return(distributions.Beta(Concentration1, Concentration0, generator: Generator));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Beta"/> distribution for each incoming RNG <see cref="torch.Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Beta distributions.</returns>
    public IObservable<TorchSharp.Modules.Beta> Process(IObservable<Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Beta(Concentration1, Concentration0, generator: Generator);
        });
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Beta"/> distribution
    /// constructed from the configured parameters and current <see cref="Generator"/>.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Beta distributions.</returns>
    public IObservable<TorchSharp.Modules.Beta> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Beta(Concentration1, Concentration0, generator: Generator));
    }
}