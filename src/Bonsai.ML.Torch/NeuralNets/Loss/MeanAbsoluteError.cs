using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates an L1 loss (L1Loss) module, known as mean absolute error (MAE).
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html"/> for more information.
/// </remarks>
[Description("Creates an L1 loss (L1Loss) module, known as mean absolute error (MAE).")]
public class MeanAbsoluteError
{
    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Creates an L1 loss (L1Loss) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(L1Loss(Reduction));
    }

    /// <summary>
    /// Creates an L1 loss (L1Loss) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => L1Loss(Reduction));
    }
}