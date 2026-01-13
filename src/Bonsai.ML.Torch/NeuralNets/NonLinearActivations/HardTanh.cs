using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.NonLinearActivations;

/// <summary>
/// Represents an operator that creates a Hardtanh module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html"/> for more information.
/// </remarks>
[Description("Creates a Hardtanh module.")]
public class HardTanh
{
    /// <summary>
    /// The minimum value of the linear region range.
    /// </summary>
    [Description("The minimum value of the linear region range.")]
    public double MinVal { get; set; } = -1D;

    /// <summary>
    /// The maximum value of the linear region range.
    /// </summary>
    [Description("The maximum value of the linear region range.")]
    public double MaxVal { get; set; } = 1D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Creates a Hardtanh module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Hardtanh(MinVal, MaxVal, Inplace));
    }

    /// <summary>
    /// Creates a Hardtanh module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Hardtanh(MinVal, MaxVal, Inplace));
    }
}