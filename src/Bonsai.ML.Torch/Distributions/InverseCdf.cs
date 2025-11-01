using Bonsai;
using System;
using System.Reactive.Linq;
using System.ComponentModel;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates an inverse cumulative distribution function (inverse CDF) from the input distribution.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates an inverse cumulative distribution function (inverse CDF) from the input distribution.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class InverseCdf : TensorContainerBase
{
    /// <summary>
    /// Initializes a new instance of the <see cref="InverseCdf"/> class.
    /// </summary>
    public InverseCdf()
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
    public Distribution Distribution { get; set; }

    /// <summary>
    /// Processes the input distribution to compute the inverse CDF at the specified values.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Distribution> source)
    {
        return source.Select(distribution => distribution.icdf(Values));
    }

    /// <summary>
    /// Processes the input values to compute the inverse CDF using the specified distribution.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(Distribution.icdf);
    }

    /// <summary>
    /// Processes the input tuples of distribution and values to compute the inverse CDF.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Distribution, Tensor>> source)
    {
        return source.Select((input) => input.Item1.icdf(input.Item2));
    }

    /// <summary>
    /// Processes the input tuples of values and distribution to compute the inverse CDF.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Distribution>> source)
    {
        return source.Select((input) => input.Item2.icdf(input.Item1));
    }
}