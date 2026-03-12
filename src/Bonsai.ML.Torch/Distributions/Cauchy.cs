using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Represents an operator that creates a Cauchy (Lorentz) distribution parameterized by location and scale.
/// </summary>
[Combinator]
[Description("Creates a Cauchy distribution with the specified location and scale parameters.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class Cauchy : IScalarTypeProvider
{
    /// <summary>
    /// The location parameter. Can be a scalar or tensor; shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The location parameter. Can be a scalar or tensor; supports batching.")]
    public Tensor Locations { get; set; } = null;

    /// <summary>
    /// The values of the locations in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Locations))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string LocationsXml
    {
        get => TensorConverter.ConvertToString(Locations, Type);
        set => Locations = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Scale parameter (> 0). Can be a scalar or tensor; must be broadcastable with <see cref="Locations"/>.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Scale parameter (> 0). Can be a scalar or tensor; must broadcast with Locations.")]
    public Tensor Scales { get; set; } = null;

    /// <summary>
    /// The values of the scales in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Scales))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ScalesXml
    {
        get => TensorConverter.ConvertToString(Scales, Type);
        set => Scales = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Gets or sets the data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Cauchy"/> distribution using the configured parameters.
    /// </summary>
    /// <returns>An observable that emits the constructed Cauchy distribution.</returns>
    public IObservable<TorchSharp.Modules.Cauchy> Process()
    {
        return Observable.Return(distributions.Cauchy(Locations, Scales));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Cauchy"/> distribution for each incoming RNG <see cref="Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Cauchy distributions.</returns>
    public IObservable<TorchSharp.Modules.Cauchy> Process(IObservable<Generator> source)
    {
        return source.Select(generator => distributions.Cauchy(Locations, Scales, generator: generator));
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Cauchy"/> distribution
    /// constructed from the configured parameters.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Cauchy distributions.</returns>
    public IObservable<TorchSharp.Modules.Cauchy> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Cauchy(Locations, Scales));
    }
}
