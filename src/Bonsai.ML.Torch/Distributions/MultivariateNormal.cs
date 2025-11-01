using static TorchSharp.torch;
using TorchSharp;
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a Multivariate Normal (Gaussian) distribution parameterized by mean and covariance matrix.
/// Emits a TorchSharp distribution module that can be sampled or queried for probabilities.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a Multivariate Normal distribution with mean vector and covariance matrix.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MultivariateNormal : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MultivariateNormal"/> class.
    /// </summary>
    public MultivariateNormal()
    {
        RegisterTensor(
            () => _mean,
            value => _mean = value);

        RegisterTensor(
            () => _covariance,
            value => _covariance = value);
    }

    private Tensor _mean;
    /// <summary>
    /// Mean vector of the distribution. Can be a 1D vector or higher-rank tensor for batched distributions.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Mean vector of the distribution. Can be a 1D vector or higher-rank tensor for batched distributions.")]
    public Tensor Mean
    {
        get => _mean;
        set => _mean = value;
    }

    /// <summary>
    /// The values of the means in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Mean))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string MeanXml
    {
        get => TensorConverter.ConvertToString(_mean, Type);
        set => _mean = TensorConverter.ConvertFromString(value, Type);
    }

    private Tensor _covariance;
    /// <summary>
    /// Covariance matrix of the distribution. Must be positive-definite and square with dimension matching <see cref="Mean"/>.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Covariance matrix of the distribution. Must be positive-definite and square with dimension matching Mean.")]
    public Tensor Covariance
    {
        get => _covariance;
        set => _covariance = value;
    }

    /// <summary>
    /// The values of the covariance matrix in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Covariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string CovarianceXml
    {
        get => TensorConverter.ConvertToString(_covariance, Type);
        set => _covariance = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Optional random number generator to use when sampling. If null, TorchSharp's global RNG is used.
    /// </summary>
    [XmlIgnore]
    public Generator Generator { get; set; } = null;

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.MultivariateNormal"/> distribution using the configured parameters and optional <see cref="Generator"/>.
    /// </summary>
    /// <returns>An observable that emits the constructed Multivariate Normal distribution.</returns>
    public IObservable<TorchSharp.Modules.MultivariateNormal> Process()
    {
        return Observable.Return(distributions.MultivariateNormal(Mean, Covariance, generator: Generator));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.MultivariateNormal"/> distribution for each incoming RNG <see cref="torch.Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Multivariate Normal distributions.</returns>
    public IObservable<TorchSharp.Modules.MultivariateNormal> Process(IObservable<Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.MultivariateNormal(Mean, Covariance, generator: Generator);
        });
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.MultivariateNormal"/> distribution
    /// constructed from the configured parameters and current <see cref="Generator"/>.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Multivariate Normal distributions.</returns>
    public IObservable<TorchSharp.Modules.MultivariateNormal> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.MultivariateNormal(Mean, Covariance, generator: Generator));
    }
}