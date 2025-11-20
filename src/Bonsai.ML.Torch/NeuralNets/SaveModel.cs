using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Saves the model to a file.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Saves the model to a file.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class SaveModel
{
    /// <summary>
    /// The model to save.
    /// </summary>
    [Description("The model to save.")]
    [XmlIgnore]
    public nn.Module Model { get; set; }

    /// <summary>
    /// The path to save the model.
    /// </summary>
    [Description("The path to save the model.")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string ModelPath { get; set; }

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
            Model.save(ModelPath);
        });
    }
}