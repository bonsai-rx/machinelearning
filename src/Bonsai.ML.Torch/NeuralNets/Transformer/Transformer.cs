using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Transformer;

/// <summary>
/// Represents an operator that creates a Transformer module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html"/> for more information.
/// </remarks>
[Description("Creates a Transformer module.")]
public class Transformer
{
    /// <summary>
    /// The number of expected features in the encoder/decoder inputs.
    /// </summary>
    [Description("The number of expected features in the encoder/decoder inputs.")]
    public long DimModel { get; set; } = 512;

    /// <summary>
    /// The number of heads in the multi-head attention models.
    /// </summary>
    [Description("The number of heads in the multi-head attention models.")]
    public long NumHeads { get; set; } = 8;

    /// <summary>
    /// The number of sub-encoder layers in the encoder.
    /// </summary>
    [Description("The number of sub-encoder layers in the encoder.")]
    public long NumEncoderLayers { get; set; } = 6;

    /// <summary>
    /// The number of sub-decoder layers in the decoder.
    /// </summary>
    [Description("The number of sub-decoder layers in the decoder.")]
    public long NumDecoderLayers { get; set; } = 6;

    /// <summary>
    /// The dimension of the feedforward network model.
    /// </summary>
    [Description("The dimension of the feedforward network model.")]
    public long DimFeedforward { get; set; } = 2048;

    /// <summary>
    /// The probability of dropout.
    /// </summary>
    [Description("The probability of dropout.")]
    public double Dropout { get; set; } = 0.1D;

    /// <summary>
    /// The activation function of the intermediate layer.
    /// </summary>
    [Description("The activation function of the intermediate layer.")]
    public Activations Activation { get; set; } = Activations.ReLU;

    /// <summary>
    /// Creates a Transformer module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(nn.Transformer(DimModel, NumHeads, NumEncoderLayers, NumDecoderLayers, DimFeedforward, Dropout, Activation));
    }

    /// <summary>
    /// Creates a Transformer module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => nn.Transformer(DimModel, NumHeads, NumEncoderLayers, NumDecoderLayers, DimFeedforward, Dropout, Activation));
    }
}
