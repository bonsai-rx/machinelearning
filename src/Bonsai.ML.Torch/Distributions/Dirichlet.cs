using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Represents an operator that creates a Dirichlet probability distribution parameterized by concentration parameters.
/// </summary>
[Combinator]
[Description("Creates a Dirichlet distribution with concentration parameters.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class Dirichlet : IScalarTypeProvider
{
    /// <summary>
    /// Concentration parameters (> 0). Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Concentration parameters (> 0). Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
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
    /// Gets or sets the data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Dirichlet"/> distribution using the configured parameters.
    /// </summary>
    /// <returns>An observable that emits the constructed Dirichlet distribution.</returns>
    public IObservable<TorchSharp.Modules.Dirichlet> Process()
    {
        return Observable.Return(distributions.Dirichlet(Concentration));
    }

    /// <summary>
    /// Creates a <see cref="TorchSharp.Modules.Dirichlet"/> distribution for each incoming RNG <see cref="Generator"/>.
    /// </summary>
    /// <param name="source">Observable sequence of random generators to use.</param>
    /// <returns>An observable sequence of Dirichlet distributions.</returns>
    public IObservable<TorchSharp.Modules.Dirichlet> Process(IObservable<Generator> source)
    {
        return source.Select(generator => distributions.Dirichlet(Concentration, generator: generator));
    }

    /// <summary>
    /// For each element of the source stream, emits a <see cref="TorchSharp.Modules.Dirichlet"/> distribution constructed from the configured parameters.
    /// </summary>
    /// <typeparam name="T">The type of the triggering source sequence.</typeparam>
    /// <param name="source">Trigger sequence; each element causes a new distribution to be emitted.</param>
    /// <returns>An observable sequence of Dirichlet distributions.</returns>
    public IObservable<TorchSharp.Modules.Dirichlet> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Dirichlet(Concentration));
    }
}