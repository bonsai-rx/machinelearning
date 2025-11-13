using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that performs matrix multiplication with 2 or more tensors.
/// </summary>
[Combinator]
[Description("Performs matrix multiplication with 2 or more tensors.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class MatrixMultiply
{
    /// <summary>
    /// Performs matrix multiplication with 2 tensors.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(input =>
        {
            return matmul(input.Item1, input.Item2);
        });
    }

    /// <summary>
    /// Performs matrix multiplication with 3 tensors.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor>> source)
    {
        return source.Select(input =>
        {
            return multi_dot([input.Item1, input.Item2, input.Item3]);
        });
    }

    /// <summary>
    /// Performs matrix multiplication with 4 tensors.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(input =>
        {
            return multi_dot([input.Item1, input.Item2, input.Item3, input.Item4]);
        });
    }

    /// <summary>
    /// Performs matrix multiplication with 5 tensors.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(input =>
        {
            return multi_dot([input.Item1, input.Item2, input.Item3, input.Item4, input.Item5]);
        });
    }

    /// <summary>
    /// Performs matrix multiplication with 6 tensors.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(input =>
        {
            return multi_dot([input.Item1, input.Item2, input.Item3, input.Item4, input.Item5, input.Item6]);
        });
    }

    /// <summary>
    /// Performs matrix multiplication with 7 tensors.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(input =>
        {
            return multi_dot([input.Item1, input.Item2, input.Item3, input.Item4, input.Item5, input.Item6, input.Item7]);
        });
    }

    /// <summary>
    /// Performs matrix multiplication with an array of tensors.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor[]> source)
    {
        return source.Select(multi_dot);
    }

    /// <summary>
    /// Performs matrix multiplication with a list of tensors.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<IList<Tensor>> source)
    {
        return source.Select(multi_dot);
    }

    /// <summary>
    /// Performs matrix multiplication with an enumerable of tensors.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<IEnumerable<Tensor>> source)
    {
        return source.Select(input => multi_dot([.. input]));
    }
}