using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Represents an operator that creates a poisson probability distribution parameterized by rate (expected number of events).
/// </summary>
[Combinator]
[Description("Creates a poisson distribution with the specified rate parameter.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class Poisson : IScalarTypeProvider
{
    /// <summary>
    /// Rate parameter (> 0), representing the expected number of events. Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Rate parameter (> 0), expected number of events. Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
    public Tensor Rate { get; set; } = null;

    /// <summary>
    /// The values of the rates in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Rate))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string RateXml
    {
        get => TensorConverter.ConvertToString(Rate, Type);
        set => Rate = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Gets or sets the data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Poisson"/> distribution using the configured parameters.
    /// </summary>
    /// <returns>An observable that emits the constructed Poisson distribution.</returns>
    public IObservable<TorchSharp.Modules.Poisson> Process()
    {
        return Observable.Return(distributions.Poisson(Rate));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Poisson"/> distribution for each incoming RNG <see cref="Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Poisson distributions.</returns>
    public IObservable<TorchSharp.Modules.Poisson> Process(IObservable<Generator> source)
    {
        return source.Select(generator => distributions.Poisson(Rate, generator: generator));
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Poisson"/> distribution constructed from the configured parameters.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Poisson distributions.</returns>
    public IObservable<TorchSharp.Modules.Poisson> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Poisson(Rate));
    }
}