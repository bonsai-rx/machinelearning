using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Represents an operator that creates a Bernoulli probability distribution parameterized by event probabilities.
/// </summary>
[Combinator]
[Description("Creates a Bernoulli distribution with event probabilities and emits a TorchSharp distribution module.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class Bernoulli : IScalarTypeProvider
{
    /// <summary>
    /// Event probabilities p in [0, 1]. Can be a scalar or a tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Event probabilities p in [0, 1]. Can be a scalar or a tensor; shape sets the batch/event shape of the distribution.")]
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
    /// Creates a Bernoulli distribution.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Bernoulli> Process()
    {
        return Observable.Return(distributions.Bernoulli(Probabilities));
    }

    /// <summary>
    /// Creates a Bernoulli distribution using the incoming RNG Generator.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Bernoulli> Process(IObservable<Generator> source)
    {
        return source.Select(generator => distributions.Bernoulli(Probabilities, generator: generator));
    }

    /// <summary>
    /// For each element of the source stream, emits a Bernoulli distribution.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Bernoulli> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Bernoulli(Probabilities));
    }
}