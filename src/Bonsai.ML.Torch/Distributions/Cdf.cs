using Bonsai;
using System;
using System.Reactive.Linq;
using System.ComponentModel;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a cumulative distribution function (CDF) from the input distribution.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a cumulative distribution function (CDF) from the input distribution.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Cdf : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Cdf"/> class.
    /// </summary>
    public Cdf()
    {
        RegisterTensor(
            () => _values,
            v => _values = v);
    }

    private Tensor _values;
    /// <summary>
    /// The values at which to evaluate the CDF.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The values at which to evaluate the CDF.")]
    public Tensor Values
    {
        get => _values;
        set => _values = value;
    }

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
    /// The input distribution.
    /// </summary>
    [XmlIgnore]
    [Description("The input distribution.")]
    public Distribution Distribution { get; set; }

    /// <summary>
    /// Processes the input distribution to compute the CDF at the specified values.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Distribution> source)
    {
        return source.Select(distribution => distribution.cdf(Values));
    }

    /// <summary>
    /// Processes the input values to compute the CDF using the specified distribution.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(Distribution.cdf);
    }

    /// <summary>
    /// Processes the input tuples of distribution and values to compute the CDF.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Distribution, Tensor>> source)
    {
        return source.Select((input) => input.Item1.cdf(input.Item2));
    }

    /// <summary>
    /// Processes the input tuples of values and distribution to compute the CDF.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Distribution>> source)
    {
        return source.Select((input) => input.Item2.cdf(input.Item1));
    }
}