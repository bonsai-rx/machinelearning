using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Module;

/// <summary>
/// Creates a 1D adaptive average pooling layer.
/// </summary>
[Combinator]
[Description("Creates a 1D adaptive average pooling layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Padding
{
    /// <summary>
    /// The number of dimensions for the AdaptiveAvgPool module.
    /// </summary>
    [Description("The number of dimensions for the AdaptiveAvgPool module")]
    public Dimensions Dimensions { get; set; } = Dimensions.One;

    /// <summary>
    /// The padding mode for the ConstantPad module.
    /// </summary>
    [Description("The padding mode for the ConstantPad module")]
    public PaddingMode Mode { get; set; } = PaddingMode.Constant;

    /// <summary>
    /// The padding parameter for the ConstantPad1d module.
    /// </summary>
    [Description("The padding parameter for the ConstantPad1d module")]
    public long PaddingSize { get; set; }

    /// <summary>
    /// The value parameter for the ConstantPad1d module.
    /// </summary>
    [Description("The value parameter for the ConstantPad1d module")]
    public double Value { get; set; }

    /// <summary>
    /// If true, uses reflection padding instead of constant padding.
    /// </summary>
    
    public bool Reflection { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a AdaptiveAvgPool1dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Dimensions switch
        {
            Dimensions.One => Mode switch
                {
                    PaddingMode.Constant => Observable.Return(ConstantPad1d(PaddingSize, Value)),
                    PaddingMode.Reflection => Observable.Return(ReflectionPad1d(PaddingSize)),
                    PaddingMode.Replication => Observable.Return(ReplicationPad1d(PaddingSize)),
                    _ => throw new InvalidOperationException("The specified padding mode is not supported."),
                },
            Dimensions.Two => Mode switch
                {
                    PaddingMode.Constant => Observable.Return(ConstantPad2d(PaddingSize, Value)),
                    PaddingMode.Reflection => Observable.Return(ReflectionPad2d(PaddingSize)),
                    PaddingMode.Replication => Observable.Return(ReplicationPad2d(PaddingSize)),
                    _ => throw new InvalidOperationException("The specified padding mode is not supported."),
                },
            Dimensions.Three => Mode switch
                {
                    PaddingMode.Constant => Observable.Return(ConstantPad3d(PaddingSize, Value)),
                    PaddingMode.Reflection => Observable.Return(ReflectionPad3d(PaddingSize)),
                    PaddingMode.Replication => Observable.Return(ReplicationPad3d(PaddingSize)),
                    _ => throw new InvalidOperationException("The specified padding mode is not supported."),
                },
            _ => throw new InvalidOperationException("The specified number of dimensions is not supported."),
        };
    }
}
