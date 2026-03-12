using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a Binomial probability distribution with a given number of trials and success probability.
/// </summary>
[Combinator]
[Description("Creates a Binomial distribution with count (number of trials) and probability of success.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class Binomial : IScalarTypeProvider
{
    /// <summary>
    /// The number of trials (non-negative). Can be a scalar or tensor. If it is a tensor, values should be non-negative integers.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The number of trials (non-negative). Can be a scalar or tensor. If it is a tensor, values should be non-negative integers.")]
    public Tensor Count { get; set; } = null;

    /// <summary>
    /// The values of count in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Count))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string CountXml
    {
        get => TensorConverter.ConvertToString(Count, Type);
        set => Count = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Probability of success p in [0, 1]. Can be a scalar or tensor; the shape should be broadcastable to <see cref="Count"/>.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Probability of success in [0, 1]. Can be a scalar or tensor; the shape should be broadcastable to Count.")]
    public Tensor Probabilities { get; set; } = null;

    /// <summary>
    /// The values of probabilities in XML string format.
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
    /// Creates a Binomial distribution.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Binomial> Process()
    {
        return Observable.Return(distributions.Binomial(Count, Probabilities));
    }

    /// <summary>
    /// Creates a Binomial distribution for each incoming RNG generator.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Binomial> Process(IObservable<Generator> source)
    {
        return source.Select(generator => distributions.Binomial(Count, Probabilities, generator: generator));
    }

    /// <summary>
    /// For each element of the source stream, emits a Binomial distribution.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Binomial> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Binomial(Count, Probabilities));
    }
}
