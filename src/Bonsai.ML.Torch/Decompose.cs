using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;

namespace Bonsai.ML.Torch;

/// <summary>
/// This operator decomposes each incoming tensor into a sequence of tensors by splitting it along the specified dimension.
/// </summary>
[Combinator]
[Description("Decomposes each incoming tensor into a sequence of tensors by splitting it along the specified dimension.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class Decompose
{
    private int _dimension = 0;
    /// <summary>
    /// Gets or sets the dimension along which to split the tensor.
    /// </summary>
    [Description("The dimension along which to split the tensor.")]
    public int Dimension
    {
        get => _dimension;
        set => _dimension = value;
    }

    /// <summary>
    /// Processes an observable sequence of tensors, decomposing each tensor into a sequence of tensors by splitting along the specified dimension.
    /// </summary>
    public IObservable<torch.Tensor> Process(IObservable<torch.Tensor> source)
    {
        return source.SelectMany((input) =>
        {
            if (input is null) return null;
            return input.unbind(_dimension).ToObservable();
        });
    }
}