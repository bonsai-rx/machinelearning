using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizer;

/// <summary>
/// Represents an operator that creates a limited-memory Broyden-Fletcher-Goldfarb-Shanno (LBFGS) optimizer.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html"/> for more information.
/// </remarks>
[Description("Creates a limited-memory Broyden-Fletcher-Goldfarb-Shanno (LBFGS) optimizer.")]
public class Lbfgs
{
    /// <summary>
    /// The learning rate.
    /// </summary>
    [Description("The learning rate.")]
    public double LearningRate { get; set; } = 0.01D;

    /// <summary>
    /// The maximum number of iterations per optimization step.
    /// </summary>
    [Description("The maximum number of iterations per optimization step.")]
    public long MaxIter { get; set; } = 20;

    /// <summary>
    /// The maximum number of function evaluations per optimization step.
    /// </summary>
    [Description("The maximum number of function evaluations per optimization step.")]
    public long? MaxEval { get; set; } = null;

    /// <summary>
    /// The termination criterion that determines when to stop optimizing based on first-order optimality.
    /// </summary>
    [Description("The termination criterion that determines when to stop optimizing based on first-order optimality.")]
    public double ToleranceGrad { get; set; } = 1E-05D;

    /// <summary>
    /// The termination criterion that determines when to stop optimizing based on function value or parameter changes.
    /// </summary>
    [Description("The termination criterion that determines when to stop optimizing based on function value or parameter changes.")]
    public double ToleranceChange { get; set; } = 1E-09D;

    /// <summary>
    /// The update history size.
    /// </summary>
    [Description("The update history size.")]
    public long HistorySize { get; set; } = 100;

    /// <summary>
    /// Creates an LBFGS optimizer from the input parameter collection.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<optim.Optimizer> Process<T>(IObservable<T> source) where T : IEnumerable<Parameter>
    {
        return source.Select(parameters => LBFGS(parameters, LearningRate, MaxIter, MaxEval, ToleranceGrad, ToleranceChange, HistorySize));
    }
}
