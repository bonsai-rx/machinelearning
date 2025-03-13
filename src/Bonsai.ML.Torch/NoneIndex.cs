using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;

namespace Bonsai.ML.Torch;

/// <summary>
/// Represents an index that selects no elements of a tensor.
/// </summary>
[Combinator]
[Description("Represents an index that selects no elements of a tensor.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class NoneIndex
{
    /// <summary>
    /// Generates the none index.
    /// </summary>
    /// <returns></returns>
    public IObservable<torch.TensorIndex> Process()
    {
        return Observable.Return(torch.TensorIndex.None);
    }

    /// <summary>
    /// Processes the input sequence and generates the none index.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<torch.TensorIndex> Process<T>(IObservable<T> source)
    {
        return source.Select((_) => torch.TensorIndex.None);
    }
}