using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.OtherModules;

/// <summary>
/// Creates a MaxUnpool1d module.
/// </summary>
[Combinator]
[Description("Creates a MaxUnpool1d module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MaxUnpool1dModule
{
    /// <summary>
    /// The kernelsize parameter for the MaxUnpool1d module.
    /// </summary>
    [Description("The kernelsize parameter for the MaxUnpool1d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride parameter for the MaxUnpool1d module.
    /// </summary>
    [Description("The stride parameter for the MaxUnpool1d module")]
    public long? Stride { get; set; } = null;

    /// <summary>
    /// The padding parameter for the MaxUnpool1d module.
    /// </summary>
    [Description("The padding parameter for the MaxUnpool1d module")]
    public long? Padding { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a MaxUnpool1dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, long[], Tensor>> Process()
    {
        return Observable.Return(MaxUnpool1d(KernelSize, Stride, Padding));
    }
}
