using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Modules;

/// <summary>
/// Creates a ELU activation function.
/// </summary>
[Combinator]
[Description("Creates a ELU activation function.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ELUModule
{
    /// <summary>
    /// The alpha parameter for the ELU module.
    /// </summary>
    [Description("The alpha parameter for the ELU module")]
    public double Alpha { get; set; } = 1D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a ELUModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(ELU(Alpha, Inplace));
    }
}
