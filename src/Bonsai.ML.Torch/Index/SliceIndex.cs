using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;

namespace Bonsai.ML.Torch.Index;

/// <summary>
/// Represents an index that selects a range of elements from a tensor.
/// </summary>
[Combinator]
[Description("Represents an index that selects a range of elements from a tensor.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SliceIndex
{
    /// <summary>
    /// Gets or sets the start index of the slice.
    /// </summary>
    [Description("The start index of the slice.")]
    public long? Start { get; set; } = null;

    /// <summary>
    /// Gets or sets the end index of the slice.
    /// </summary>
    [Description("The end index of the slice.")]
    public long? End { get; set; } = null;

    /// <summary>
    /// Gets or sets the step size of the slice.
    /// </summary>
    [Description("The step size of the slice.")]
    public long? Step { get; set; } = null;

    /// <summary>
    /// Generates the slice index.
    /// </summary>
    /// <returns></returns>
    public IObservable<torch.TensorIndex> Process()
    {
        return Observable.Return(torch.TensorIndex.Slice(Start, End, Step));
    }

    /// <summary>
    /// Processes the input sequence and generates the slice index.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<torch.TensorIndex> Process<T>(IObservable<T> source)
    {
        return source.Select((_) => torch.TensorIndex.Slice(Start, End, Step));
    }
}