using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Represents an operator that fits a PCA model to the input data and transforms it.
/// </summary>
[Combinator]
[Description("Fits a PCA model to the input data and transforms it.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class FitAndTransform
{
    private static Tensor FitModelAndTransformData(PcaBaseModel model, Tensor data)
    {
        return model.FitAndTransform(data);
    }

    /// <summary>
    /// Fits a PCA model to the input data and transforms it.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<T, Tensor>> Process<T>(IObservable<Tuple<T, Tensor>> source) where T : PcaBaseModel
    {
        return source.Select((value) =>
        {
            var transformed = FitModelAndTransformData(value.Item1, value.Item2);
            return Tuple.Create(value.Item1, transformed);
        });
    }

    /// <summary>
    /// Fits a PCA model to the input data and transforms it.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Tensor, T>> Process<T>(IObservable<Tuple<Tensor, T>> source) where T : PcaBaseModel
    {
        return source.Select((value) =>
        {
            var transformed = FitModelAndTransformData(value.Item2, value.Item1);
            return Tuple.Create(transformed, value.Item2);
        });
    }
}
