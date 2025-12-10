using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Represents an operator that creates a multivariate normal (Gaussian) distribution parameterized by mean vector and covariance matrix.
/// </summary>
[Combinator]
[Description("Creates a multivariate normal (Gaussian) distribution with mean vector and covariance matrix.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class MultivariateNormal : IScalarTypeProvider
{
    /// <summary>
    /// Mean vector of the distribution. Can be a 1D vector or higher-rank tensor for batched distributions.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Mean vector of the distribution. Can be a 1D vector or higher-rank tensor for batched distributions.")]
    public Tensor Mean { get; set; } = null;

    /// <summary>
    /// The values of the means in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Mean))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string MeanXml
    {
        get => TensorConverter.ConvertToString(Mean, Type);
        set => Mean = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Covariance matrix of the distribution. Must be positive-definite and square with dimension matching <see cref="Mean"/>.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Covariance matrix of the distribution. Must be positive-definite and square with dimension matching Mean.")]
    public Tensor Covariance { get; set; } = null;

    /// <summary>
    /// The values of the covariance matrix in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Covariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string CovarianceXml
    {
        get => TensorConverter.ConvertToString(Covariance, Type);
        set => Covariance = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Gets or sets the data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.MultivariateNormal"/> distribution using the configured parameters.
    /// </summary>
    /// <returns>An observable that emits the constructed Multivariate Normal distribution.</returns>
    public IObservable<TorchSharp.Modules.MultivariateNormal> Process()
    {
        return Observable.Return(distributions.MultivariateNormal(Mean, Covariance));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.MultivariateNormal"/> distribution for each incoming RNG <see cref="Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Multivariate Normal distributions.</returns>
    public IObservable<TorchSharp.Modules.MultivariateNormal> Process(IObservable<Generator> source)
    {
        return source.Select(generator => distributions.MultivariateNormal(Mean, Covariance, generator: generator));
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.MultivariateNormal"/> distribution constructed from the configured parameters.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Multivariate Normal distributions.</returns>
    public IObservable<TorchSharp.Modules.MultivariateNormal> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.MultivariateNormal(Mean, Covariance));
    }
}