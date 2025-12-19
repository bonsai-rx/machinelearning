using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;

namespace Bonsai.ML.Torch;

/// <summary>
/// Represents an operator that deconstructs each tensor in the sequence into one or more tensors by splitting it along the specified dimension.
/// </summary>
[Combinator]
[Description("Deconstructs each tensor in the sequence into one or more tensors by splitting it along the specified dimension.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class Unbind
{
    private int _dimension = 0;
    /// <summary>
    /// Gets or sets the dimension along which to deconstruct the tensor.
    /// </summary>
    [Description("The dimension along which to deconstruct the tensor.")]
    public int Dimension
    {
        get => _dimension;
        set => _dimension = value;
    }

    /// <summary>
    /// Processes an observable sequence of tensors, deconstructing each tensor into a sequence of tensors by splitting along the specified dimension.
    /// </summary>
    public IObservable<torch.Tensor> Process(IObservable<torch.Tensor> source)
    {
        return source.SelectMany((input) =>
        {
            if (input is null) 
                return null;
            return input.unbind(_dimension).ToObservable();
        });
    }
}