using Bonsai;
using System;
using System.Reactive.Linq;
using System.ComponentModel;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Generates samples from the input distribution.
/// Gradients do not flow through the sampling process.
/// </summary>
[Combinator]
[Description("Generates samples from the input distribution.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Sample
{
    /// <summary>
    /// The shape of the samples to generate.
    /// </summary>
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] SampleShape { get; set; }

    /// <summary>
    /// Processes the input distribution to generate samples.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Distribution> source)
    {
        return source.Select(distribution => distribution.sample(SampleShape));
    }
}