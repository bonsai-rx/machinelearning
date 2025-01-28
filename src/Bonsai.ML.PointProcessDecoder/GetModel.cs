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
[WorkflowElementCategory(ElementCategory.Source)]
[Description("Returns the point process model.")]
public class GetModel
{
    /// <summary>
    /// The name of the point process model to return.
    /// </summary>
    [TypeConverter(typeof(PointProcessModelNameConverter))]
    public string Model { get; set; } = string.Empty;

    /// <summary>
    /// Returns the point process model.
    /// </summary>
    /// <returns></returns>
    public IObservable<PointProcessModel> Process()
    {
        return Observable.Defer(() => 
            Observable.Return(PointProcessModelManager.GetModel(Model))
        );
    }

    /// <summary>
    /// Returns the point process model.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<PointProcessModel> Process<T>(IObservable<T> source)
    {
        return source.Select(input => {
            return PointProcessModelManager.GetModel(Model);
        });
    }
}