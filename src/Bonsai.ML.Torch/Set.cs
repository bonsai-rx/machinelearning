using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

/// <summary>
/// Sets values of a tensor to the provided values at the specified index.
/// </summary>
[Combinator]
[Description("Sets the value of the input tensor at the specified index.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Set
{
    /// <summary>
    /// The index at which to set the value.
    /// </summary>
    [Description("The index at which to set the value.")]
    public string Index { get; set; } = string.Empty;

    /// <summary>
    /// The value to set at the specified index.
    /// </summary>
    [XmlIgnore]
    [Description("The value to set at the specified index.")]
    public Tensor Value { get; set; } = null;

    /// <summary>
    /// Returns an observable sequence that sets the value of the input tensor to the `Value` property at indices specified by the `Index` property.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(tensor => tensor.index_put_(Value, IndexHelper.Parse(Index)));
    }

    /// <summary>
    /// Returns an observable sequence that sets the value of the input tensor to the `Value` property using the tensor index in the input sequence.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, torch.TensorIndex>> source)
    {
        return source.Select(input => input.Item1.index_put_(Value, input.Item2));
    }

    /// <summary>
    /// Returns an observable sequence that sets the value of the input tensor to the `Value` property using the tensor index array in the input sequence.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, torch.TensorIndex[]>> source)
    {
        return source.Select(input => input.Item1.index_put_(Value, input.Item2));
    }

    /// <summary>
    /// Returns an observable sequence that sets the value of the input tensor in item 1 to item 2 at indices specified by the `Index` property.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(input => input.Item1.index_put_(input.Item2, IndexHelper.Parse(Index)));
    }

    /// <summary>
    /// Returns an observable sequence that sets the value of the input tensor in item 1 to item 2 at the tensor index specified in item 3.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, torch.TensorIndex>> source)
    {
        return source.Select(input => input.Item1.index_put_(input.Item2, input.Item3));
    }

    /// <summary>
    /// Returns an observable sequence that sets the value of the input tensor in item 1 to item 2 at the tensor index array specified in item 3.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, torch.TensorIndex[]>> source)
    {
        return source.Select(input => input.Item1.index_put_(input.Item2, input.Item3));
    }
}