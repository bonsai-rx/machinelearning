using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;

namespace Bonsai.ML.Torch;

/// <summary>
/// Represents an operator that returns a view of each tensor in the sequence as a tensor containing all slices of a specified size and step along a given dimension. 
/// </summary>
[Combinator]
[Description("Returns a view of each tensor in the sequence as a tensor containing all slices of a specified size and step along a given dimension.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Unfold
{
    /// <summary>
    /// Gets or sets the dimension along which unfolding happens.
    /// </summary>
    [Description("The dimension along which unfolding happens.")]
    public int Dimension { get; set; } = 0;

    /// <summary>
    /// Gets or sets the size of each slice that is unfolded.
    /// </summary>
    [Description("The size of each slice that is unfolded.")]
    public int Size { get; set; } = 1;

    /// <summary>
    /// Gets or sets the step between each slice.
    /// </summary>
    [Description("The step between each slice.")]
    public int Step { get; set; } = 1;

    /// <summary>
    /// Processes an observable sequence of tensors, deconstructing each tensor into a sequence of tensors by splitting along the specified dimension.
    /// </summary>
    public IObservable<torch.Tensor> Process(IObservable<torch.Tensor> source)
    {
        return source.Select((input) =>
        {
            if (input is null) 
                return null;
            return input.unfold(Dimension, Size, Step);
        });
    }
}