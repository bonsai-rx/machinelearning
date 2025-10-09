using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Losses;

/// <summary>
/// Creates a CTCLoss module.
/// </summary>
[Combinator]
[Description("Creates a CTCLoss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CTCLoss
{
    /// <summary>
    /// The blank parameter for the CTCLoss module.
    /// </summary>
    [Description("The blank parameter for the CTCLoss module")]
    public long Blank { get; set; } = 0;

    /// <summary>
    /// The zero_infinity parameter for the CTCLoss module.
    /// </summary>
    [Description("The zero_infinity parameter for the CTCLoss module")]
    public bool ZeroInfinity { get; set; } = false;

    /// <summary>
    /// The reduction parameter for the CTCLoss module.
    /// </summary>
    [Description("The reduction parameter for the CTCLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a CTCLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(CTCLoss(Blank, ZeroInfinity, Reduction));
    }
}
