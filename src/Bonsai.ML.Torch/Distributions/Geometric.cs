using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Represents an operator that creates a geometric probability distribution parameterized by success probability.
/// </summary>
[Combinator]
[Description("Creates a geometric distribution with the specified success probability.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class Geometric : IScalarTypeProvider
{
    /// <summary>
    /// Success probability in [0, 1]. Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Success probability in [0, 1]. Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
    public Tensor Probabilities { get; set; } = null;

    /// <summary>
    /// The values of the probabilities in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Probabilities))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ProbabilitiesXml
    {
        get => TensorConverter.ConvertToString(Probabilities, Type);
        set => Probabilities = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Gets or sets the data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Geometric"/> distribution using the configured parameters.
    /// </summary>
    /// <returns>An observable that emits the constructed Geometric distribution.</returns>
    public IObservable<TorchSharp.Modules.Geometric> Process()
    {
        return Observable.Return(distributions.Geometric(Probabilities));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Geometric"/> distribution for each incoming RNG <see cref="Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Geometric distributions.</returns>
    public IObservable<TorchSharp.Modules.Geometric> Process(IObservable<Generator> source)
    {
        return source.Select(generator => distributions.Geometric(Probabilities, generator: generator));
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Geometric"/> distribution
    /// constructed from the configured parameters.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Geometric distributions.</returns>
    public IObservable<TorchSharp.Modules.Geometric> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Geometric(Probabilities));
    }
}