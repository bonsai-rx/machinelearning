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
/// Sets the training mode for the model.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Sets the training mode for the model.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class SetTrainingMode
{
    /// <summary>
    /// The model for which to set the training mode. 
    /// </summary>
    [Description("The model for which to set the training mode.")]
    [XmlIgnore]
    public IModule<Tensor, Tensor> Model { get; set; }

    /// <summary>
    /// The training mode to set for the model.
    /// </summary>
    [Description("The training mode to set for the model.")]
    public TrainingMode Mode { get; set; } = TrainingMode.Train;

    /// <summary>
    /// Saves the model to the specified file path.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<T> Process<T>(IObservable<T> source)
    {
        return source.Do(input =>
        {

            var training = Mode == TrainingMode.Train;

            switch (Model)
            {
                case Module<Tensor, Tensor> module:
                    module.train(training);
                    break;
                case ScriptModule<Tensor, Tensor> scriptModule:
                    scriptModule.train(training);
                    break;
                default:
                    throw new InvalidOperationException("Unsupported model type for setting training mode.");
            }
        });
    }
}