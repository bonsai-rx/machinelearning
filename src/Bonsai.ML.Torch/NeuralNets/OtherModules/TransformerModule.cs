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
/// Creates a Transformer module.
/// </summary>
[Combinator]
[Description("Creates a Transformer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class TransformerModule
{
    /// <summary>
    /// The d_model parameter for the Transformer module.
    /// </summary>
    [Description("The d_model parameter for the Transformer module")]
    public long DModel { get; set; } = 512;

    /// <summary>
    /// The nhead parameter for the Transformer module.
    /// </summary>
    [Description("The nhead parameter for the Transformer module")]
    public long Nhead { get; set; } = 8;

    /// <summary>
    /// The num_encoder_layers parameter for the Transformer module.
    /// </summary>
    [Description("The num_encoder_layers parameter for the Transformer module")]
    public long NumEncoderLayers { get; set; } = 6;

    /// <summary>
    /// The num_decoder_layers parameter for the Transformer module.
    /// </summary>
    [Description("The num_decoder_layers parameter for the Transformer module")]
    public long NumDecoderLayers { get; set; } = 6;

    /// <summary>
    /// The dim_feedforward parameter for the Transformer module.
    /// </summary>
    [Description("The dim_feedforward parameter for the Transformer module")]
    public long DimFeedforward { get; set; } = 2048;

    /// <summary>
    /// The dropout parameter for the Transformer module.
    /// </summary>
    [Description("The dropout parameter for the Transformer module")]
    public double Dropout { get; set; } = 0.1D;

    /// <summary>
    /// The activation parameter for the Transformer module.
    /// </summary>
    [Description("The activation parameter for the Transformer module")]
    public Activations Activation { get; set; } = Activations.ReLU;

    /// <summary>
    /// Generates an observable sequence that creates a TransformerModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(Transformer(DModel, Nhead, NumEncoderLayers, NumDecoderLayers, DimFeedforward, Dropout, Activation));
    }
}
