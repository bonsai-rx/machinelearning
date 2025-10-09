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
/// Creates a Hardtanh module.
/// </summary>
[Combinator]
[Description("Creates a Hardtanh module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class HardtanhModule
{
    /// <summary>
    /// The min_val parameter for the Hardtanh module.
    /// </summary>
    [Description("The min_val parameter for the Hardtanh module")]
    public double MinVal { get; set; } = -1D;

    /// <summary>
    /// The max_val parameter for the Hardtanh module.
    /// </summary>
    [Description("The max_val parameter for the Hardtanh module")]
    public double MaxVal { get; set; } = 1D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a HardtanhModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Hardtanh(MinVal, MaxVal, Inplace));
    }
}
