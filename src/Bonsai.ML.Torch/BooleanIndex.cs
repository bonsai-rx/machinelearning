using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;

namespace Bonsai.ML.Torch;

/// <summary>
/// Represents a boolean index that can be used to select elements from a tensor.
/// </summary>
[Combinator]
[Description("Represents a boolean index that can be used to select elements from a tensor.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class BooleanIndex
{
    /// <summary>
    /// Gets or sets the boolean value used to select elements from a tensor.
    /// </summary>
    [Description("The boolean value used to select elements from a tensor.")]
    public bool Value { get; set; } = false;

    /// <summary>
    /// Generates the boolean index.
    /// </summary>
    /// <returns></returns>
    public IObservable<torch.TensorIndex> Process()
    {
        return Observable.Return(torch.TensorIndex.Bool(Value));
    }
    
    /// <summary>
    /// Processes the input sequence and generates the boolean index.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<torch.TensorIndex> Process<T>(IObservable<T> source)
    {
        return source.Select((_) => torch.TensorIndex.Bool(Value));
    }
}