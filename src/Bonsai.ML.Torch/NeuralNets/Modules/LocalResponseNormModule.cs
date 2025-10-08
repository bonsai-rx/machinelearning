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
/// Creates a LocalResponseNorm module module.
/// </summary>
[Combinator]
[Description("Creates a LocalResponseNorm module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LocalResponseNormModule
{
    /// <summary>
    /// The size parameter for the LocalResponseNorm module.
    /// </summary>
    [Description("The size parameter for the LocalResponseNorm module")]
    public long Size { get; set; }

    /// <summary>
    /// The alpha parameter for the LocalResponseNorm module.
    /// </summary>
    [Description("The alpha parameter for the LocalResponseNorm module")]
    public double Alpha { get; set; } = 0.0001;

    /// <summary>
    /// The beta parameter for the LocalResponseNorm module.
    /// </summary>
    [Description("The beta parameter for the LocalResponseNorm module")]
    public double Beta { get; set; } = 0.75;

    /// <summary>
    /// The k parameter for the LocalResponseNorm module.
    /// </summary>
    [Description("The k parameter for the LocalResponseNorm module")]
    public double K { get; set; } = 1;

    /// <summary>
    /// Generates an observable sequence that creates a LocalResponseNorm module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(LocalResponseNorm(Size, Alpha, Beta, K));
    }
}
