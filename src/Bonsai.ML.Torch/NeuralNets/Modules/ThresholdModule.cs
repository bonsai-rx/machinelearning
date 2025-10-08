using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Modules;

/// <summary>
/// Creates a Threshold module module.
/// </summary>
[Combinator]
[Description("Creates a Threshold module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ThresholdModule
{
    /// <summary>
    /// The threshold parameter for the Threshold module.
    /// </summary>
    [Description("The threshold parameter for the Threshold module")]
    public double Threshold { get; set; }

    /// <summary>
    /// The value parameter for the Threshold module.
    /// </summary>
    [Description("The value parameter for the Threshold module")]
    public double Value { get; set; }

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a Threshold module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Threshold(Threshold, Value, Inplace));
    }
}
