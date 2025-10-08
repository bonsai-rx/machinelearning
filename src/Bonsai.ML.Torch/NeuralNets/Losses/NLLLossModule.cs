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
/// Creates a NLLLoss module module.
/// </summary>
[Combinator]
[Description("Creates a NLLLoss module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class NLLLossModule : IScalarTypeProvider
{
    /// <summary>
    /// The weight parameter for the NLLLoss module.
    /// </summary>
    [Description("The weight parameter for the NLLLoss module")]
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Weight { get; set; } = null;

    /// <summary>
    /// The reduction parameter for the NLLLoss module.
    /// </summary>
    [Description("The reduction parameter for the NLLLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// The scalar type for the module.
    /// </summary>
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Generates an observable sequence that creates a NLLLoss module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(NLLLoss(Weight, Reduction));
    }
}
