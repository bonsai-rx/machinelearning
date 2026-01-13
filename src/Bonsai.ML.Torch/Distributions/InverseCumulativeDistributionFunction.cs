using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates an inverse cumulative distribution function (inverse CDF) from the input distribution.
/// </summary>
[Combinator]
[Description("Creates an inverse cumulative distribution function (inverse CDF) from the input distribution.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class InverseCumulativeDistributionFunction
{
    /// <summary>
    /// The input distribution.
    /// </summary>
    [XmlIgnore]
    public Distribution Distribution { get; set; }

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