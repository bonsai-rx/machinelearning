using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;

namespace Bonsai.ML.Torch;

/// <summary>
/// Represents an index that selects a single value of a tensor.
/// </summary>
[Combinator]
[Description("Represents an index that selects a single value of a tensor.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SingleIndex
{
    /// <summary>
    /// Gets or sets the index value used to select a single element from a tensor.
    /// </summary>
    [Description("The index value used to select a single element from a tensor.")]
    public long Value { get; set; } = 0;

    /// <summary>
    /// Generates the single index.
    /// </summary>
    /// <returns></returns>
    public IObservable<torch.TensorIndex> Process()
    {
        return Observable.Return(torch.TensorIndex.Single(Value));
    }

    /// <summary>
    /// Processes the input sequence and generates the single index.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<torch.TensorIndex> Process<T>(IObservable<T> source)
    {
        return source.Select((_) => torch.TensorIndex.Single(Value));
    }
}