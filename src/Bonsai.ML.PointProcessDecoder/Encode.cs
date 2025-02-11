using System;
using System.ComponentModel;
using System.Reactive.Linq;

using static TorchSharp.torch;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Encodes the combined state observation data and neural data into a point process model.
/// </summary>
[Combinator]
[WorkflowElementCategory(ElementCategory.Sink)]
[Description("Encodes the combined state observation data and neural data into a point process model.")]
public class Encode
{
    /// <summary>
    /// The name of the point process model to use.
    /// </summary>
    [TypeConverter(typeof(PointProcessModelNameConverter))]
    [Description("The name of the point process model to use.")]
    public string Model { get; set; } = string.Empty;

    /// <summary>
    /// Encodes the combined state observation data and neural data into a point process model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Tensor, Tensor>> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        var modelName = Model;
        return source.Do(input =>
        {
            var model = PointProcessModelManager.GetModel(modelName);
            var (neuralData, stateObservations) = input;
            model.Encode(neuralData, stateObservations);
        });
    }
}