using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Modules;

/// <summary>
/// Creates a Fold module module.
/// </summary>
[Combinator]
[Description("Creates a Fold module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class FoldModule
{
    /// <summary>
    /// The output_size parameter for the Fold module.
    /// </summary>
    [Description("The output_size parameter for the Fold module")]
    public long OutputSize { get; set; }

    /// <summary>
    /// The kernel_size parameter for the Fold module.
    /// </summary>
    [Description("The kernel_size parameter for the Fold module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The dilation parameter for the Fold module.
    /// </summary>
    [Description("The dilation parameter for the Fold module")]
    public long Dilation { get; set; } = 1;

    /// <summary>
    /// The padding parameter for the Fold module.
    /// </summary>
    [Description("The padding parameter for the Fold module")]
    public long Padding { get; set; } = 0;

    /// <summary>
    /// The stride parameter for the Fold module.
    /// </summary>
    [Description("The stride parameter for the Fold module")]
    public long Stride { get; set; } = 1;

    /// <summary>
    /// Generates an observable sequence that creates a Fold module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Fold(OutputSize, KernelSize, Dilation, Padding, Stride));
    }
}
