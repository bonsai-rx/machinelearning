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
/// Measures the Binary Cross Entropy between the target and the output logits.
/// </summary>
[Combinator]
[Description("Measures the Binary Cross Entropy between the target and the output logits.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class BinaryCrossEntropyWithLogits : IScalarTypeProvider
{
    /// <summary>
    /// The weight parameter for the BinaryCrossEntropyWithLogitsLoss module.
    /// </summary>
    [Description("The weight parameter for the BinaryCrossEntropyWithLogitsLoss module")]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Weight { get; set; } = null;

    /// <summary>
    /// The reduction parameter for the BinaryCrossEntropyWithLogitsLoss module.
    /// </summary>
    [Description("The reduction parameter for the BinaryCrossEntropyWithLogitsLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// The pos_weights parameter for the BinaryCrossEntropyWithLogitsLoss module.
    /// </summary>
    [Description("The pos_weights parameter for the BinaryCrossEntropyWithLogitsLoss module")]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor PosWeights { get; set; } = null;

    /// <summary>
    /// The data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Generates an observable sequence that creates a BinaryCrossEntropyWithLogitsLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(BCEWithLogitsLoss(Weight, Reduction, PosWeights));
    }
}
