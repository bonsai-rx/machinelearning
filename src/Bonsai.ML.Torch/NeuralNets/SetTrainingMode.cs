using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.jit;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that sets the training mode for the module.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Sets the training mode for the module.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class SetTrainingMode
{
    /// <summary>
    /// The training mode to set for the module.
    /// </summary>
    [Description("The training mode to set for the module.")]
    public TrainingMode Mode { get; set; } = TrainingMode.Train;

    /// <summary>
    /// Sets the training mode for the module.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<nn.Module> Process(IObservable<nn.Module> source)
    {
        return source.Do(input =>
        {
            // var training = Mode == TrainingMode.Train;
            input.train(Mode == TrainingMode.Train);
        });
    }
}