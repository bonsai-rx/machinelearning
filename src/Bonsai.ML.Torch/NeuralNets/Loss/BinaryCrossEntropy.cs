using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Creates a binary cross entropy loss module.
/// </summary>
[Combinator]
[Description("Creates a binary cross entropy loss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class BinaryCrossEntropy : IScalarTypeProvider
{
    /// <summary>
    /// The weight parameter for the BinaryCrossEntropy module.
    /// </summary>
    [Description("The weight parameter for the BinaryCrossEntropy module")]
    public Tensor Weight { get; set; } = null;

    /// <summary>
    /// The reduction parameter for the BinaryCrossEntropy module.
    /// </summary>
    [Description("The reduction parameter for the BinaryCrossEntropy module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// The data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Generates an observable sequence that creates a BinaryCrossEntropy.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(BCELoss(Weight, Reduction));
    }
}
