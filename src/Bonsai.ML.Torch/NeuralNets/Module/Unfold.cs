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
/// Creates a Unfold module.
/// </summary>
[Combinator]
[Description("Creates a Unfold module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Unfold
{
    /// <summary>
    /// The kernel_size parameter for the Unfold module.
    /// </summary>
    [Description("The kernel_size parameter for the Unfold module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The dilation parameter for the Unfold module.
    /// </summary>
    [Description("The dilation parameter for the Unfold module")]
    public long Dilation { get; set; } = 1;

    /// <summary>
    /// The padding parameter for the Unfold module.
    /// </summary>
    [Description("The padding parameter for the Unfold module")]
    public long Padding { get; set; } = 0;

    /// <summary>
    /// The stride parameter for the Unfold module.
    /// </summary>
    [Description("The stride parameter for the Unfold module")]
    public long Stride { get; set; } = 1;

    /// <summary>
    /// Generates an observable sequence that creates a UnfoldModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Unfold(KernelSize, Dilation, Padding, Stride));
    }
}
