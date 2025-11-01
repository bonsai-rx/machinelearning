using static TorchSharp.torch;
using TorchSharp;
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a Dirichlet probability distribution parameterized by concentration parameters.
/// Emits a TorchSharp distribution module that can be sampled or queried for probabilities.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a Dirichlet distribution with concentration parameters.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Dirichlet : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Dirichlet"/> class.
    /// </summary>
    public Dirichlet()
    {
        RegisterTensor(
            () => _concentration,
            value => _concentration = value);
    }

    private Tensor _concentration;
    /// <summary>
    /// Concentration parameters (> 0). Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [TypeConverter(typeof(TensorConverter))]
    [Description("Concentration parameters (> 0). Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
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

    /// <summary>
    /// Optional random number generator to use when sampling. If null, TorchSharp's global RNG is used.
    /// </summary>
    [XmlIgnore]
    public Generator Generator { get; set; } = null;

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Dirichlet"/> distribution using the configured parameters and optional <see cref="Generator"/>.
    /// </summary>
    /// <returns>An observable that emits the constructed Dirichlet distribution.</returns>
    public IObservable<TorchSharp.Modules.Dirichlet> Process()
    {
        return Observable.Return(distributions.Dirichlet(Concentration, generator: Generator));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Dirichlet"/> distribution for each incoming RNG <see cref="torch.Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Dirichlet distributions.</returns>
    public IObservable<TorchSharp.Modules.Dirichlet> Process(IObservable<Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Dirichlet(Concentration, generator: Generator);
        });
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Dirichlet"/> distribution
    /// constructed from the configured parameters and current <see cref="Generator"/>.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Dirichlet distributions.</returns>
    public IObservable<TorchSharp.Modules.Dirichlet> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Dirichlet(Concentration, generator: Generator));
    }
}