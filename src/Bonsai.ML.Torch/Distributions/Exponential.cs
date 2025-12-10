using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Represents an operator that creates an exponential probability distribution parameterized by rate.
/// </summary>
[Combinator]
[Description("Creates an exponential distribution with the specified rate parameter.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class Exponential : IScalarTypeProvider
{
    /// <summary>
    /// Rate parameter (> 0). Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Rate parameter (> 0). Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
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
    /// Creates a <see cref="TorchSharp.Modules.Exponential"/> distribution using the configured parameters.
    /// </summary>
    /// <returns>An observable that emits the constructed Exponential distribution.</returns>
    public IObservable<TorchSharp.Modules.Exponential> Process()
    {
        return Observable.Return(distributions.Exponential(Rate));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Exponential"/> distribution for each incoming RNG <see cref="Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Exponential distributions.</returns>
    public IObservable<TorchSharp.Modules.Exponential> Process(IObservable<Generator> source)
    {
        return source.Select(generator => distributions.Exponential(Rate, generator: generator));
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Exponential"/> distribution constructed from the configured parameters.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Exponential distributions.</returns>
    public IObservable<TorchSharp.Modules.Exponential> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Exponential(Rate));
    }
}