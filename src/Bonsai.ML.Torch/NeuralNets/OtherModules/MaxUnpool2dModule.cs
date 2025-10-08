using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.OtherModules;

/// <summary>
/// Creates a MaxUnpool2d module module.
/// </summary>
[Combinator]
[Description("Creates a MaxUnpool2d module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MaxUnpool2dModule
{
    /// <summary>
    /// The kernelsize parameter for the MaxUnpool2d module.
    /// </summary>
    [Description("The kernelsize parameter for the MaxUnpool2d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride parameter for the MaxUnpool2d module.
    /// </summary>
    [Description("The stride parameter for the MaxUnpool2d module")]
    public long? Stride { get; set; } = null;

    /// <summary>
    /// The padding parameter for the MaxUnpool2d module.
    /// </summary>
    [Description("The padding parameter for the MaxUnpool2d module")]
    public long? Padding { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a MaxUnpool2d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Int64[], Tensor>> Process()
    {
        return Observable.Return(MaxUnpool2d(KernelSize, Stride, Padding));
    }
}
