using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;

namespace Bonsai.ML.Torch;

/// <summary>
/// Represents an operator that returns the element-wise sine of each tensor in the sequence.
/// </summary>
[Combinator]
[Description("Returns the element-wise sine of each tensor in the sequence.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Sin
{
    /// <summary>
    /// Processes an observable sequence of tensors, returning the element-wise sine of each tensor.
    /// </summary>
    public IObservable<torch.Tensor> Process(IObservable<torch.Tensor> source)
    {
        return source.Select((input) =>
        {
            if (input is null) 
                return null;
            return input.sin();
        });
    }
}