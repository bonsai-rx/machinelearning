using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

/// <summary>
/// Clones the input tensor.
/// </summary>
[Combinator]
[Description("Clones the input tensor.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Clone
{
    /// <summary>
    /// Clones the input tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(tensor => tensor.clone());
    }
}