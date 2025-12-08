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
    /// Creates a TransformerDecoder module from the input TransformerDecoderLayer.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> Process(IObservable<Module<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select(transformerDecoderLayer =>
            TransformerDecoder(transformerDecoderLayer as TorchSharp.Modules.TransformerDecoderLayer, NumLayers));
    }
}
