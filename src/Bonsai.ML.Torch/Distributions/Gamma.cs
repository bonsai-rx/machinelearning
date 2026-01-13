using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Represents an operator that creates a gamma probability distribution parameterized by concentration and rate.
/// </summary>
[Combinator]
[Description("Creates a gamma distribution with concentration and rate parameters.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class Gamma : IScalarTypeProvider
{
    /// <summary>
    /// Concentration parameter (> 0). Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Concentration parameter (> 0). Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
    public Tensor Concentration { get; set; } = null;

    /// <summary>
    /// The values of the concentration in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Concentration))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ConcentrationXml
    {
        get => TensorConverter.ConvertToString(Concentration, Type);
        set => Concentration = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Rate parameter (> 0). Can be a scalar or tensor; must be broadcastable with <see cref="Concentration"/>.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Rate parameter (> 0). Can be a scalar or tensor; must broadcast with Concentration.")]
    public Tensor Rate { get; set; } = null;

    /// <summary>
    /// The values of the rate in XML string format.
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
    /// Creates a <see cref="TorchSharp.Modules.Gamma"/> distribution using the configured parameters.
    /// </summary>
    /// <returns>An observable that emits the constructed Gamma distribution.</returns>
    public IObservable<TorchSharp.Modules.Gamma> Process()
    {
        return Observable.Return(distributions.Gamma(Concentration, Rate));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Gamma"/> distribution for each incoming RNG <see cref="Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Gamma distributions.</returns>
    public IObservable<TorchSharp.Modules.Gamma> Process(IObservable<Generator> source)
    {
        return source.Select(generator => distributions.Gamma(Concentration, Rate, generator: generator));
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Gamma"/> distribution
    /// constructed from the configured parameters.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Gamma distributions.</returns>
    public IObservable<TorchSharp.Modules.Gamma> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Gamma(Concentration, Rate));
    }
}