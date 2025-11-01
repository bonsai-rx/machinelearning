using Bonsai;
using System;
using System.Reactive.Linq;
using System.ComponentModel;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Computes the log probability of the given values under the specified distribution.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Computes the log probability of the given values under the specified distribution.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class LogProbability : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LogProbability"/> class.
    /// </summary>
    public LogProbability()
    {
        RegisterTensor(
            () => _values,
            v => _values = v);
    }

    private Tensor _values;
    /// <summary>
    /// The values at which to evaluate the inverse CDF.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The values at which to evaluate the inverse CDF.")]
    public Tensor Values
    {
        get => _values;
        set => _values = value;
    }

    /// <summary>
    /// The input distribution.
    /// </summary>
    [XmlIgnore]
    public Distribution Distribution { get; set; }

    /// <summary>
    /// The values in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Values))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ValuesXml
    {
        get => TensorConverter.ConvertToString(_values, Type);
        set => _values = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Processes the input distribution to compute the log probability at the specified values.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Distribution> source)
    {
        return source.Select(distribution => distribution.log_prob(Values));
    }

    /// <summary>
    /// Processes the input values to compute the log probability using the specified distribution.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(Distribution.log_prob);
    }

    /// <summary>
    /// Processes the input tuples of distribution and values to compute the log probability.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Distribution, Tensor>> source)
    {
        return source.Select((input) => input.Item1.log_prob(input.Item2));
    }

    /// <summary>
    /// Processes the input tuples of values and distribution to compute the log probability.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Distribution>> source)
    {
        return source.Select((input) => input.Item2.log_prob(input.Item1));
    }
}