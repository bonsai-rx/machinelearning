using System;
using System.Linq;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that performs optimized matrix multiplication of 2 or more tensors so that the fewest number
/// of operations are performed.
/// </summary>
/// <remarks>
/// Every tensor in the input sequence must be 2D, except for the first or last tensor which may be 1D. If the first
/// tensor is 1D, it is treated as a row vector and if the last tensor is 1D, it is treated as a column vector. The
/// output will be 2D if both the first and last tensors are 2D, otherwise the output will be 1D.
/// </remarks>
[Combinator]
[Description("Performs optimized matrix multiplication of 2 or more tensors using the multi_dot function.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class MultiDot
{
    /// <summary>
    /// Performs optimized matrix multiplication of 2 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(value => multi_dot([value.Item1, value.Item2]));
    }

    /// <summary>
    /// Performs optimized matrix multiplication of 3 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor>> source)
    {
        return source.Select(value => multi_dot([value.Item1, value.Item2, value.Item3]));
    }

    /// <summary>
    /// Performs optimized matrix multiplication of 4 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(value => multi_dot([value.Item1, value.Item2, value.Item3, value.Item4]));
    }

    /// <summary>
    /// Performs optimized matrix multiplication of 5 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(value => multi_dot([value.Item1, value.Item2, value.Item3, value.Item4, value.Item5]));
    }

    /// <summary>
    /// Performs optimized matrix multiplication of 6 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(value => multi_dot([value.Item1, value.Item2, value.Item3, value.Item4, value.Item5, value.Item6]));
    }

    /// <summary>
    /// Performs optimized matrix multiplication of 7 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(value => multi_dot([value.Item1, value.Item2, value.Item3, value.Item4, value.Item5, value.Item6, value.Item7]));
    }

    /// <summary>
    /// Performs optimized matrix multiplication of an enumerable of tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<IEnumerable<Tensor>> source)
    {
        return source.Select(value =>
        {
            if (value.Count() < 2)
                return value.FirstOrDefault();

            return multi_dot([.. value]);
        });
    }
}
