using System;
using System.ComponentModel;
using System.Reactive.Linq;

using static TorchSharp.torch;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Decodes the input neural data into a posterior state estimate using a point process model.
/// </summary>
[Combinator]
[WorkflowElementCategory(ElementCategory.Transform)]
[Description("Decodes the input neural data into a posterior state estimate using a point process model.")]
public class Decode
{
    /// <summary>
    /// The name of the point process model to use.
    /// </summary>
    [TypeConverter(typeof(PointProcessModelNameConverter))]
    public string Model { get; set; } = string.Empty;

    /// <summary>
    /// Decodes the input neural data into a posterior state estimate using a point process model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(input => {
            var model = PointProcessModelManager.GetModel(Model);
            return model.Decode(input);
        });
    }
}