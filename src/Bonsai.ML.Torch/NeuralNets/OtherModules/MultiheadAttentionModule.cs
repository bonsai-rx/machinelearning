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
/// Creates a Multi-head attention mechanism module.
/// </summary>
[Combinator]
[Description("Creates a Multi-head attention mechanism module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MultiheadAttentionModule
{
    /// <summary>
    /// The embedded_dim parameter for the MultiheadAttention module.
    /// </summary>
    [Description("The embedded_dim parameter for the MultiheadAttention module")]
    public long EmbeddedDim { get; set; }

    /// <summary>
    /// The num_heads parameter for the MultiheadAttention module.
    /// </summary>
    [Description("The num_heads parameter for the MultiheadAttention module")]
    public long NumHeads { get; set; }

    /// <summary>
    /// The dropout parameter for the MultiheadAttention module.
    /// </summary>
    [Description("The dropout parameter for the MultiheadAttention module")]
    public double Dropout { get; set; } = 0;

    /// <summary>
    /// If true, adds a learnable bias to the output.
    /// </summary>
    [Description("If true, adds a learnable bias to the output")]
    public bool Bias { get; set; } = true;

    /// <summary>
    /// The add_bias_kv parameter for the MultiheadAttention module.
    /// </summary>
    [Description("The add_bias_kv parameter for the MultiheadAttention module")]
    public bool AddBiasKv { get; set; } = false;

    /// <summary>
    /// The add_zero_attn parameter for the MultiheadAttention module.
    /// </summary>
    [Description("The add_zero_attn parameter for the MultiheadAttention module")]
    public bool AddZeroAttn { get; set; } = false;

    /// <summary>
    /// The kdim parameter for the MultiheadAttention module.
    /// </summary>
    [Description("The kdim parameter for the MultiheadAttention module")]
    public long? Kdim { get; set; } = null;

    /// <summary>
    /// The vdim parameter for the MultiheadAttention module.
    /// </summary>
    [Description("The vdim parameter for the MultiheadAttention module")]
    public long? Vdim { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a MultiheadAttention module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor, Tensor?, bool, Tensor?, Tuple<Tensor, Tensor>>> Process()
    {
        return Observable.Return(MultiheadAttention(EmbeddedDim, NumHeads, Dropout, Bias, AddBiasKv, AddZeroAttn, Kdim, Vdim));
    }
}
