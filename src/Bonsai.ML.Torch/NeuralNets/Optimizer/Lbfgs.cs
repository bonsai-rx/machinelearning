using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizer;

/// <summary>
/// Creates a LBFGS optimizer.
/// </summary>
[Combinator]
[Description("Creates a LBFGS optimizer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Lbfgs
{
    /// <summary>
    /// The parameters parameter for the LBFGS module.
    /// </summary>
    [Description("The parameters parameter for the LBFGS module")]
    public IEnumerable<(string, Parameter)> Parameters { get; set; }

    /// <summary>
    /// The lr parameter for the LBFGS module.
    /// </summary>
    [Description("The lr parameter for the LBFGS module")]
    public double Lr { get; set; } = 0.01D;

    /// <summary>
    /// The max_iter parameter for the LBFGS module.
    /// </summary>
    [Description("The max_iter parameter for the LBFGS module")]
    public long MaxIter { get; set; } = 20;

    /// <summary>
    /// The max_eval parameter for the LBFGS module.
    /// </summary>
    [Description("The max_eval parameter for the LBFGS module")]
    public long? MaxEval { get; set; } = null;

    /// <summary>
    /// The tolerange_grad parameter for the LBFGS module.
    /// </summary>
    [Description("The tolerange_grad parameter for the LBFGS module")]
    public double TolerangeGrad { get; set; } = 1E-05D;

    /// <summary>
    /// The tolerance_change parameter for the LBFGS module.
    /// </summary>
    [Description("The tolerance_change parameter for the LBFGS module")]
    public double ToleranceChange { get; set; } = 1E-09D;

    /// <summary>
    /// The history_size parameter for the LBFGS module.
    /// </summary>
    [Description("The history_size parameter for the LBFGS module")]
    public long HistorySize { get; set; } = 100;

    /// <summary>
    /// Generates an observable sequence that creates a LBFGSOptimizer.
    /// </summary>
    public IObservable<optim.Optimizer> Process()
    {
        return Observable.Return(LBFGS(Parameters, Lr, MaxIter, MaxEval, TolerangeGrad, ToleranceChange, HistorySize));
    }
}
