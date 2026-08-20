using System;
using System.Linq;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

/// <summary>
/// Represents an operator that performs matrix multiplication of 2 or more tensors.
/// </summary>
[Combinator]
[Description("Performs matrix multiplication of 2 or more tensors.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class MatrixMultiply
{
    /// <summary>
    /// Performs matrix multiplication of 2 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(value => value.Item1.matmul(value.Item2));
    }

    /// <summary>
    /// Performs matrix multiplication of 3 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor>> source)
    {
        return source.Select(value => value.Item1.matmul(value.Item2).matmul(value.Item3));
    }

    /// <summary>
    /// Performs matrix multiplication of 4 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(value => value.Item1.matmul(value.Item2).matmul(value.Item3).matmul(value.Item4));
    }

    /// <summary>
    /// Performs matrix multiplication of 5 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(value => value.Item1.matmul(value.Item2).matmul(value.Item3).matmul(value.Item4).matmul(value.Item5));
    }

    /// <summary>
    /// Performs matrix multiplication of 6 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(value => value.Item1.matmul(value.Item2).matmul(value.Item3).matmul(value.Item4).matmul(value.Item5).matmul(value.Item6));
    }

    /// <summary>
    /// Performs matrix multiplication of 7 tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(value => value.Item1.matmul(value.Item2).matmul(value.Item3).matmul(value.Item4).matmul(value.Item5).matmul(value.Item6).matmul(value.Item7));
    }

    /// <summary>
    /// Performs matrix multiplication of an enumerable of tensors.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<IEnumerable<Tensor>> source)
    {
        return source.Select(value =>
        {
            var result = value.FirstOrDefault();
            foreach (var tensor in value.Skip(1))
            {
                result = result.matmul(tensor);
            }
            return result;
        });
    }
}
