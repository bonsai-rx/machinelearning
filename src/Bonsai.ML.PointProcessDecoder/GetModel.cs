using System;
using System.ComponentModel;
using System.Reactive.Linq;
using PointProcessDecoder.Core;
using static TorchSharp.torch;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Returns the point process model.
/// </summary>
[Combinator]
[WorkflowElementCategory(ElementCategory.Transform)]
[Description("Returns the point process model.")]
public class GetModel
{
    /// <summary>
    /// The name of the point process model to return.
    /// </summary>
    [TypeConverter(typeof(PointProcessModelNameConverter))]
    public string Model { get; set; } = string.Empty;

    /// <summary>
    /// Decodes the input neural data into a posterior state estimate using a point process model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<PointProcessModel> Process(IObservable<Tensor> source)
    {
        return source.Select(input => {
            return PointProcessModelManager.GetModel(Model);
        });
    }
}