using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Transformer;

/// <summary>
/// Represents an operator that creates a transformer decoder module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html"/> for more information.
/// </remarks>
[Description("Creates a transformer decoder module.")]
public class TransformerDecoder
{
    /// <summary>
    /// The number of sub-decoder layers in the decoder.
    /// </summary>
    [Description("The number of sub-decoder layers in the decoder.")]
    public long NumLayers { get; set; }

    /// <summary>
    /// The transformer decoder layer to be stacked.
    /// </summary>
    [XmlIgnore]
    [Description("The transformer decoder layer to be stacked.")]
    public TorchSharp.Modules.TransformerDecoderLayer TransformerDecoderLayer { get; private set; }

    /// <summary>
    /// Creates a TransformerDecoder module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(TransformerDecoder(TransformerDecoderLayer, NumLayers));
    }
}
