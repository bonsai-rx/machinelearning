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
/// Prints the string representation of incoming tensors to the console.
/// </summary>
[Combinator]
[Description("Prints the string representation of incoming tensors to the console.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class PrintTensor
{
    /// <summary>
    /// Gets or sets the string style used to format the tensor output.
    /// </summary>
    [Description("The string style used to format the tensor output.")]
    public TensorStringStyle StringStyle { get; set; }

    /// <summary>
    /// Processes the input sequence of tensors and prints their string representations to the console.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Do(value => Console.WriteLine(value.ToString(StringStyle)));
    }
}
