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
/// Creates a TransformerDecoder module module.
/// </summary>
[Combinator]
[Description("Creates a TransformerDecoder module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class TransformerDecoderModule
{
    /// <summary>
    /// The decoder_layer parameter for the TransformerDecoder module.
    /// </summary>
    [Description("The decoder_layer parameter for the TransformerDecoder module")]
    public TransformerDecoderLayer DecoderLayer { get; set; }

    /// <summary>
    /// The num_layers parameter for the TransformerDecoder module.
    /// </summary>
    [Description("The num_layers parameter for the TransformerDecoder module")]
    public long NumLayers { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a TransformerDecoder module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(TransformerDecoder(DecoderLayer, NumLayers));
    }
}
