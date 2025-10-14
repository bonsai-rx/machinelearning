using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Module;

/// <summary>
/// Creates a GELU activation function.
/// </summary>
[Combinator]
[Description("Creates a GELU activation function.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class GaussianErrorLinearUnit
{
    /// <summary>
    /// The inPlace parameter for the GELU module.
    /// </summary>
    [Description("The inPlace parameter for the GELU module")]
    public bool InPlace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a GaussianErrorLinearUnit module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(GELU(InPlace));
    }
}
