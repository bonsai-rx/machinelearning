using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Linq;

namespace Bonsai.ML.Torch.NeuralNets.Container;

/// <summary>
/// Represents an operator that creates a sequential container.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html"/> for more information.
/// </remarks>
[Description("Creates a sequential container.")]
public class Sequential
{
    /// <summary>
    /// Creates a sequential container from the input modules.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Sequential> Process<T>(IObservable<T[]> source) where T : Module<Tensor, Tensor>
    {
        return source.Select(modules => Sequential(modules));
    }
}
