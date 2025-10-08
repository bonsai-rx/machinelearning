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
/// Creates a MaxUnpool3d module module.
/// </summary>
[Combinator]
[Description("Creates a MaxUnpool3d module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MaxUnpool3dModule
{
    /// <summary>
    /// The kernelsize parameter for the MaxUnpool3d module.
    /// </summary>
    [Description("The kernelsize parameter for the MaxUnpool3d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride parameter for the MaxUnpool3d module.
    /// </summary>
    [Description("The stride parameter for the MaxUnpool3d module")]
    public long? Stride { get; set; } = null;

    /// <summary>
    /// The padding parameter for the MaxUnpool3d module.
    /// </summary>
    [Description("The padding parameter for the MaxUnpool3d module")]
    public long? Padding { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a MaxUnpool3d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Int64[], Tensor>> Process()
    {
        return Observable.Return(MaxUnpool3d(KernelSize, Stride, Padding));
    }
}
