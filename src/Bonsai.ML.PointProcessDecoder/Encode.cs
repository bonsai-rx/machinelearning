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
[Description("Encodes the combined state observation data and neural data into a point process model. The input should be a tuple of (covariate, observation) tensors. Covariate tensors should have shape (numSamples, covariateDim). Spike observations should have shape (num_samples, num_units). Clusterless marks should have shape (numSamples, markDim, numChannels).")]
public class Encode : IPointProcessModelReference
{
    /// <summary>
    /// The name of the point process model to use.
    /// </summary>
    [TypeConverter(typeof(PointProcessModelNameConverter))]
    [Description("The name of the point process model to use.")]
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Encodes the combined state observation data and neural data into a point process model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Tensor, Tensor>> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        var modelName = Name;
        return source.Do(input =>
        {
            var model = PointProcessModelManager.GetModel(modelName);
            var (covariates, observations) = input;
            model.Encode(covariates, observations);
        });
    }
}
