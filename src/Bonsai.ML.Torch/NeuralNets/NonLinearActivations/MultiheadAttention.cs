using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.NonLinearActivations;

/// <summary>
/// Represents an operator that creates a multi-head attention mechanism.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html"/> for more information.
/// </remarks>
[Description("Creates a multi-head attention mechanism.")]
public class MultiheadAttention
{
    /// <summary>
    /// The dimension of the model.
    /// </summary>
    [Description("The dimension of the model.")]
    public long EmbeddedDim { get; set; }

    /// <summary>
    /// The number of parallel attention heads.
    /// </summary>
    [Description("The number of parallel attention heads.")]
    public long NumHeads { get; set; }

    /// <summary>
    /// The dropout probability on attended weights.
    /// </summary>
    [Description("The dropout probability on attended weights.")]
    public double Dropout { get; set; } = 0D;

    /// <summary>
    /// If true, adds a learnable bias to the input/output projection layers.
    /// </summary>
    [Description("If true, adds a learnable bias to the input/output projection layers.")]
    public bool Bias { get; set; } = true;

    /// <summary>
    /// If true, adds bias to the key and value sequences at dimension 0.
    /// </summary>
    [Description("If true, adds bias to the key and value sequences at dimension 0.")]
    public bool AddBiasKeyValue { get; set; } = false;

    /// <summary>
    /// If true, adds a new batch of zeros to the key and value sequences at dimension 1.
    /// </summary>
    [Description("If true, adds a new batch of zeros to the key and value sequences at dimension 1.")]
    public bool AddZeroAttention { get; set; } = false;

    /// <summary>
    /// The total number of features for keys.
    /// </summary>
    [Description("The total number of features for keys.")]
    public long? KeyDim { get; set; } = null;

    /// <summary>
    /// The total number of features for values.
    /// </summary>
    [Description("The total number of features for values.")]
    public long? ValueDim { get; set; } = null;

    /// <summary>
    /// Creates a Multi-head attention module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor, Tensor, bool, Tensor, Tuple<Tensor, Tensor>>> Process()
    {
        return Observable.Return(MultiheadAttention(EmbeddedDim, NumHeads, Dropout, Bias, AddBiasKeyValue, AddZeroAttention, KeyDim, ValueDim));
    }

    /// <summary>
    /// Creates a Multi-head attention module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor, Tensor, bool, Tensor, Tuple<Tensor, Tensor>>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => MultiheadAttention(EmbeddedDim, NumHeads, Dropout, Bias, AddBiasKeyValue, AddZeroAttention, KeyDim, ValueDim));
    }
}
