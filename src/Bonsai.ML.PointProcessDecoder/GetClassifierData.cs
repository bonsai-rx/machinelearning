using System;
using System.ComponentModel;
using System.Reactive.Linq;
using PointProcessDecoder.Core.Decoder;
using static TorchSharp.torch;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Transforms the output of a classifier model into a <see cref="ClassifierData"/> object.
/// </summary>
[Combinator]
[WorkflowElementCategory(ElementCategory.Transform)]
[Description("Transforms the output of a decoder model into a ClassifierData object.")]
public class GetClassifierData : IPointProcessModelReference
{
    /// <summary>
    /// The name of the point process model to use.
    /// </summary>
    [TypeConverter(typeof(PointProcessModelNameConverter))]
    [Description("The name of the point process model to use.")]
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Converts the input tensor representing the decoded output of a point process classifier model into a packaged <see cref="ClassifierData"/> object.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<ClassifierData> Process(IObservable<Tensor> source)
    {
        var modelName = Name;
        return source.Select(input => 
        {
            var model = PointProcessModelManager.GetModel(modelName);
            return new ClassifierData(model.StateSpace, input);
        });
    }
}