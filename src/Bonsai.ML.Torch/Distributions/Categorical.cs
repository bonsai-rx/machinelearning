using static TorchSharp.torch;
using TorchSharp;
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a Categorical (discrete) distribution over classes given event probabilities.
/// Emits a TorchSharp distribution module that can be sampled or queried for probabilities.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a Categorical distribution with class probabilities and emits a TorchSharp distribution module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Categorical : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Categorical"/> class.
    /// </summary>
    public Categorical()
    {
        RegisterTensor(
            () => _probabilities,
            value => _probabilities = value);
    }

    private Tensor _probabilities;
    /// <summary>
    /// Class probabilities along the last dimension. Values must be non-negative and typically sum to 1 per row.
    /// Can be a 1D vector or higher-rank tensor for batched distributions.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Class probabilities along the last dimension (non-negative, typically sum to 1 per row). Supports batching.")]
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
    /// Creates a <see cref="TorchSharp.Modules.Categorical"/> distribution using the configured <see cref="Probabilities"/>.
    /// </summary>
    /// <returns>An observable that emits the constructed Categorical distribution.</returns>
    public IObservable<TorchSharp.Modules.Categorical> Process()
    {
        return Observable.Return(distributions.Categorical(Probabilities, generator: Generator));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Categorical"/> distribution for each incoming RNG <see cref="torch.Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Categorical distributions.</returns>
    public IObservable<TorchSharp.Modules.Categorical> Process(IObservable<torch.Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.Categorical(Probabilities, generator: Generator);
        });
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Categorical"/> distribution
    /// constructed from the configured <see cref="Probabilities"/> and current <see cref="Generator"/>.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Categorical distributions.</returns>
    public IObservable<TorchSharp.Modules.Categorical> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Categorical(Probabilities, generator: Generator));
    }
}