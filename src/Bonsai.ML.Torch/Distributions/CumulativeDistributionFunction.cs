using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Represents an operator that creates a cumulative distribution function (CDF) from the given distribution.
/// </summary>
[Combinator]
[Description("Creates a cumulative distribution function (CDF) from the given distribution.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class CumulativeDistributionFunction
{
    /// <summary>
    /// The input distribution.
    /// </summary>
    [XmlIgnore]
    [Description("The input distribution.")]
    public Distribution Distribution { get; set; }

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