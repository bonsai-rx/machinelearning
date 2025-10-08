using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Losses;

/// <summary>
/// Creates a MSELoss module module.
/// </summary>
[Combinator]
[Description("Creates a MSELoss module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MSELossModule
{
    /// <summary>
    /// The reduction parameter for the MSELoss module.
    /// </summary>
    [Description("The reduction parameter for the MSELoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a MSELoss module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(MSELoss(Reduction));
    }
}
