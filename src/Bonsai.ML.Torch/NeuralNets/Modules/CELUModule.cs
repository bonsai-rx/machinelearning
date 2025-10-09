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
/// Creates a CELU module.
/// </summary>
[Combinator]
[Description("Creates a CELU module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CELUModule
{
    /// <summary>
    /// The alpha parameter for the CELU module.
    /// </summary>
    [Description("The alpha parameter for the CELU module")]
    public double Alpha { get; set; } = 1D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a CELUModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(CELU(Alpha, Inplace));
    }
}
