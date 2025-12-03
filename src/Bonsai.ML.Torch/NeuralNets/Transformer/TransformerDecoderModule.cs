using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Transformer;

/// <summary>
/// Creates a TransformerDecoder module.
/// </summary>
[Combinator]
[Description("Creates a TransformerDecoder module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class TransformerDecoderModule
{
    /// <summary>
    /// The decoder_layer parameter for the TransformerDecoder module.
    /// </summary>
    [Description("The decoder_layer parameter for the TransformerDecoder module")]
    [XmlIgnore]
    public Module<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> DecoderLayer { get; set; }

    /// <summary>
    /// The num_layers parameter for the TransformerDecoder module.
    /// </summary>
    [Description("The num_layers parameter for the TransformerDecoder module")]
    public long NumLayers { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a TransformerDecoderModule module.
    /// </summary>
    public IObservable<Module<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> Process()
    {

        return Observable.Return(TransformerDecoder(DecoderLayer as TorchSharp.Modules.TransformerDecoderLayer, NumLayers));
    }
}
