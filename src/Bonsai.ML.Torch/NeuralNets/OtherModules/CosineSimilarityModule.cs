using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.OtherModules;

/// <summary>
/// Creates a CosineSimilarity module module.
/// </summary>
[Combinator]
[Description("Creates a CosineSimilarity module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CosineSimilarityModule
{
    /// <summary>
    /// The dim parameter for the CosineSimilarity module.
    /// </summary>
    [Description("The dim parameter for the CosineSimilarity module")]
    public long Dim { get; set; } = 1;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-08;

    /// <summary>
    /// Generates an observable sequence that creates a CosineSimilarity module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(CosineSimilarity(Dim, Eps));
    }
}
