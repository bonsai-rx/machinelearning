using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

/// <summary>
/// Indexes a tensor by parsing the specified indices. Indices are specified as a comma-separated values. 
/// Currently supports Python-style slicing syntax, which includes numerical indices, None, slices, and ellipsis.
/// </summary>
[Combinator]
[Description("Indexes a tensor by parsing the specified indices. Indices are specified as a comma-separated values. Currently supports Python-style slicing syntax, which includes numerical indices, None, slices, and ellipsis.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Index
{
    /// <summary>
    /// The indices to use for indexing the tensor.
    /// </summary>
    [Description("The indices to use for indexing the tensor. For example, '...,3:5,:'")]
    public string Value { get; set; } = string.Empty;

    /// <summary>
    /// Indexes the input tensor with the specified indices.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        var index = IndexHelper.Parse(Value);
        return source.Select(tensor =>  tensor.index(index));
    }

    /// <summary>
    /// Indexes the input tensor with the specified indices.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, torch.TensorIndex>> source)
    {
        return source.Select(value => value.Item1.index(new torch.TensorIndex[] { value.Item2 }));
    }

    /// <summary>
    /// Indexes the input tensor with the specified indices.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, IEnumerable<torch.TensorIndex>>> source)
    {
        return source.Select(value => value.Item1.index(value.Item2.ToArray()));
    }
}
