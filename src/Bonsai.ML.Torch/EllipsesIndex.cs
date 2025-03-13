using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;

namespace Bonsai.ML.Torch;

/// <summary>
/// Represents an index that selects all dimensions of a tensor.
/// </summary>
[Combinator]
[Description("Represents an index that selects all dimensions of a tensor.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class EllipsisIndex
{
    /// <summary>
    /// Generates the ellipsis index.
    /// </summary>
    /// <returns></returns>
    public IObservable<torch.TensorIndex> Process()
    {
        return Observable.Return(torch.TensorIndex.Ellipsis);
    }

    /// <summary>
    /// Processes the input sequence and generates the ellipsis index.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<torch.TensorIndex> Process<T>(IObservable<T> source)
    {
        return source.Select((_) => torch.TensorIndex.Ellipsis);
    }
}