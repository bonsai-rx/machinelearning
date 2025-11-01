using Bonsai;
using System;
using System.Reactive.Linq;
using System.ComponentModel;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Generates reparameterized samples from the input distribution.
/// Reparameterized samples allow gradients to flow through the sampling process.
/// </summary>
[Combinator]
[Description("Generates reparameterized samples from the input distribution.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class ReparametrizedSample
{
    /// <summary>
    /// The shape of the samples to generate.
    /// </summary>
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] SampleShape { get; set; }

    /// <summary>
    /// Processes the input distribution to generate reparameterized samples.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Distribution> source)
    {
        return source.Select(distribution => distribution.rsample(SampleShape));
    }
}