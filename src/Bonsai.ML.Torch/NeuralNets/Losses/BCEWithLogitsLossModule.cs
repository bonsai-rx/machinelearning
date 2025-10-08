using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Losses;

/// <summary>
/// Creates a BCEWithLogitsLoss module module.
/// </summary>
[Combinator]
[Description("Creates a BCEWithLogitsLoss module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class BCEWithLogitsLossModule : IScalarTypeProvider
{
    /// <summary>
    /// The weight parameter for the BCEWithLogitsLoss module.
    /// </summary>
    [Description("The weight parameter for the BCEWithLogitsLoss module")]
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Weight { get; set; } = null;

    /// <summary>
    /// The reduction parameter for the BCEWithLogitsLoss module.
    /// </summary>
    [Description("The reduction parameter for the BCEWithLogitsLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// The pos_weights parameter for the BCEWithLogitsLoss module.
    /// </summary>
    [Description("The pos_weights parameter for the BCEWithLogitsLoss module")]
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor PosWeights { get; set; } = null;

    /// <summary>
    /// The scalar type for the module.
    /// </summary>
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Generates an observable sequence that creates a BCEWithLogitsLoss module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(BCEWithLogitsLoss(Weight, Reduction, PosWeights));
    }
}
