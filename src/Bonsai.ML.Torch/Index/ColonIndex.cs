using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;

namespace Bonsai.ML.Torch.Index;

/// <summary>
/// Represents the colon index used to select all elements along a given dimension.
/// </summary>
[Combinator]
[Description("Represents the colon index used to select all elements along a given dimension.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ColonIndex
{
    /// <summary>
    /// Generates the colon index.
    /// </summary>
    /// <returns></returns>
    public IObservable<torch.TensorIndex> Process()
    {
        return Observable.Return(torch.TensorIndex.Colon);
    }

    /// <summary>
    /// Processes the input sequence and generates the colon index.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<torch.TensorIndex> Process<T>(IObservable<T> source)
    {
        return source.Select((_) => torch.TensorIndex.Colon);
    }
}