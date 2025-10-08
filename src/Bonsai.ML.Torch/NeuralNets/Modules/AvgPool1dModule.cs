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
/// Creates a 1D average pooling layer module.
/// </summary>
[Combinator]
[Description("Creates a 1D average pooling layer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class AvgPool1dModule
{
    /// <summary>
    /// The kernel_size parameter for the AvgPool1d module.
    /// </summary>
    [Description("The kernel_size parameter for the AvgPool1d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride parameter for the AvgPool1d module.
    /// </summary>
    [Description("The stride parameter for the AvgPool1d module")]
    public long? Stride { get; set; } = null;

    /// <summary>
    /// The padding parameter for the AvgPool1d module.
    /// </summary>
    [Description("The padding parameter for the AvgPool1d module")]
    public long Padding { get; set; } = 0;

    /// <summary>
    /// The ceil_mode parameter for the AvgPool1d module.
    /// </summary>
    [Description("The ceil_mode parameter for the AvgPool1d module")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// The count_include_pad parameter for the AvgPool1d module.
    /// </summary>
    [Description("The count_include_pad parameter for the AvgPool1d module")]
    public bool CountIncludePad { get; set; } = true;

    /// <summary>
    /// Generates an observable sequence that creates a AvgPool1d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(AvgPool1d(KernelSize, Stride, Padding, CeilMode, CountIncludePad));
    }
}
