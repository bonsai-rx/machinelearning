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
/// Creates a PoissonNLLLoss module.
/// </summary>
[Combinator]
[Description("Creates a PoissonNLLLoss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PoissonNLLLoss
{
    /// <summary>
    /// The log_input parameter for the PoissonNLLLoss module.
    /// </summary>
    [Description("The log_input parameter for the PoissonNLLLoss module")]
    public bool LogInput { get; set; } = true;

    /// <summary>
    /// The full parameter for the PoissonNLLLoss module.
    /// </summary>
    [Description("The full parameter for the PoissonNLLLoss module")]
    public bool Full { get; set; } = false;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public float Eps { get; set; } = 1E-08F;

    /// <summary>
    /// The reduction parameter for the PoissonNLLLoss module.
    /// </summary>
    [Description("The reduction parameter for the PoissonNLLLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a PoissonNLLLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(PoissonNLLLoss(LogInput, Full, Eps, Reduction));
    }
}
