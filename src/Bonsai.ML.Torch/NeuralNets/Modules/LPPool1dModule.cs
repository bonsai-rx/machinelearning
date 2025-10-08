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
/// Creates a LPPool1d module module.
/// </summary>
[Combinator]
[Description("Creates a LPPool1d module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LPPool1dModule
{
    /// <summary>
    /// The norm_type parameter for the LPPool1d module.
    /// </summary>
    [Description("The norm_type parameter for the LPPool1d module")]
    public double NormType { get; set; }

    /// <summary>
    /// The kernelsize parameter for the LPPool1d module.
    /// </summary>
    [Description("The kernelsize parameter for the LPPool1d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride parameter for the LPPool1d module.
    /// </summary>
    [Description("The stride parameter for the LPPool1d module")]
    public long? Stride { get; set; } = null;

    /// <summary>
    /// The ceil_mode parameter for the LPPool1d module.
    /// </summary>
    [Description("The ceil_mode parameter for the LPPool1d module")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a LPPool1d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(LPPool1d(NormType, KernelSize, Stride, CeilMode));
    }
}
