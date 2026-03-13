using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Transformer;

/// <summary>
/// Represents an operator that creates a transformer encoder module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html"/> for more information.
/// </remarks>
[Description("Creates a transformer encoder module.")]
public class TransformerEncoder
{
    /// <summary>
    /// The number of sub-encoder layers in the encoder.
    /// </summary>
    [Description("The number of sub-encoder layers in the encoder.")]
    public long NumLayers { get; set; }

    /// <summary>
    /// Creates a TransformerEncoder module from the input TransformerEncoderLayer.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.TransformerEncoder> Process(IObservable<TorchSharp.Modules.TransformerEncoderLayer> source)
    {
        return source.Select(input => TransformerEncoder(input, NumLayers));
    }
}
