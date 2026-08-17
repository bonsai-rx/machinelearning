using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Transformer;

/// <summary>
/// Represents an operator that creates a transformer encoder layer.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html"/> for more information.
/// </remarks>
[Description("Creates a transformer encoder layer.")]
public class TransformerEncoderLayer
{
    /// <summary>
    /// The number of expected features in the input.
    /// </summary>
    [Description("The number of expected features in the input.")]
    public long DimModel { get; set; } = 512;

    /// <summary>
    /// The number of heads in the multi-head attention models.
    /// </summary>
    [Description("The number of heads in the multi-head attention models.")]
    public long NumHeads { get; set; } = 8;

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
    /// Creates a TransformerEncoderLayer module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.TransformerEncoderLayer> Process()
    {
        return Observable.Return(TransformerEncoderLayer(DimModel, NumHeads, DimFeedforward, Dropout, Activation));
    }

    /// <summary>
    /// Creates a TransformerEncoderLayer module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.TransformerEncoderLayer> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => TransformerEncoderLayer(DimModel, NumHeads, DimFeedforward, Dropout, Activation));
    }
}
