using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the solution X to the system tensordot(A, X) = B.
/// </summary>
[Combinator]
[Description("Computes the solution to the system tensordot(A, X) = B.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class TensorSolve
{
    /// <summary>
    /// The dimension to perform the operation.
    /// </summary>
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] Dimensions { get; set; } = [];

    /// <summary>
    /// Computes the cross product of 2 tensors.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(value =>
        {
            return tensorsolve(value.Item1, value.Item2, Dimensions);
        });
    }
}