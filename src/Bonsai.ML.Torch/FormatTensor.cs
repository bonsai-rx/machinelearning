using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

/// <summary>
/// Represents an operator that applies a string formatting operation to all tensors in the sequence.
/// </summary>
[Combinator]
[Description("Applies a string formatting operation to all tensors in the sequence.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class FormatTensor
{
    /// <summary>
    /// Gets or sets the string style used to format the tensor output.
    /// </summary>
    [Description("The string style used to format the tensor output.")]
    public TensorStringStyle StringStyle { get; set; }

    /// <summary>
    /// Applies a string formatting operation to all tensors in an observable sequence.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<string> Process(IObservable<Tensor> source)
    {
        return source.Select(value => value.ToString(StringStyle));
    }
}
