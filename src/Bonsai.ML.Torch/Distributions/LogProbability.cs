using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Represents an operator that computes the log probability of the input values under the specified distribution.
/// </summary>
[Combinator]
[Description("Computes the log probability of the input values under the specified distribution.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class LogProbability
{
    /// <summary>
    /// The input distribution.
    /// </summary>
    [XmlIgnore]
    public Distribution Distribution { get; set; }

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