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
/// Creates a TransformerDecoderLayer module.
/// </summary>
[Combinator]
[Description("Creates a TransformerDecoderLayer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class TransformerDecoderLayerModule
{
    /// <summary>
    /// The d_model parameter for the TransformerDecoderLayer module.
    /// </summary>
    [Description("The d_model parameter for the TransformerDecoderLayer module")]
    public long DModel { get; set; } = 512;

    /// <summary>
    /// The nhead parameter for the TransformerDecoderLayer module.
    /// </summary>
    [Description("The nhead parameter for the TransformerDecoderLayer module")]
    public long Nhead { get; set; } = 8;

    /// <summary>
    /// The dim_feedforward parameter for the TransformerDecoderLayer module.
    /// </summary>
    [Description("The dim_feedforward parameter for the TransformerDecoderLayer module")]
    public long DimFeedforward { get; set; } = 2048;

    /// <summary>
    /// The dropout parameter for the TransformerDecoderLayer module.
    /// </summary>
    [Description("The dropout parameter for the TransformerDecoderLayer module")]
    public double Dropout { get; set; } = 0.1D;

    /// <summary>
    /// The activation parameter for the TransformerDecoderLayer module.
    /// </summary>
    [Description("The activation parameter for the TransformerDecoderLayer module")]
    public Activations Activation { get; set; } = Activations.ReLU;

    /// <summary>
    /// Generates an observable sequence that creates a TransformerDecoderLayerModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(TransformerDecoderLayer(DModel, Nhead, DimFeedforward, Dropout, Activation));
    }
}
