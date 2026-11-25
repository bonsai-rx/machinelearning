using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the cross product of 2 tensors.
/// </summary>
[Combinator]
[Description("Computes the cross product of 2 tensors.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class CrossProduct
{
    /// <summary>
    /// Gets or sets the dimension along which to compute the cross product.
    /// </summary>
    [Description("The dimension along which to compute the cross product.")]
    public long Dimension { get; set; } = -1;

    /// <summary>
    /// Computes the cross product of 2 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(value => cross(value.Item1, value.Item2, Dimension));
    }
}
