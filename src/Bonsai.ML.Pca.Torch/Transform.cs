using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Represents an operator that transforms input data using a PCA model.
/// </summary>
[Combinator]
[Description("Transforms input data using a PCA model.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Transform
{
    private static Tensor TransformData(PcaBaseModel model, Tensor data)
    {
        return model.Transform(data);
    }

    /// <summary>
    /// Transforms input data using a PCA model.
    /// </summary>
    /// <typeparam name="T">The type of the PCA model.</typeparam>
    /// <param name="source">
    /// A sequence of pairs of a PCA model and the data to transform, a 2D tensor with shape
    /// (samples x features).
    /// </param>
    /// <returns>
    /// A sequence of the transformed data, 2D tensors with shape (samples x components).
    /// </returns>
    public IObservable<Tensor> Process<T>(IObservable<Tuple<T, Tensor>> source) where T : PcaBaseModel
    {
        return source.Select(value =>
        {
            return TransformData(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Transforms input data using a PCA model.
    /// </summary>
    /// <typeparam name="T">The type of the PCA model.</typeparam>
    /// <param name="source">
    /// A sequence of pairs of a PCA model and the data to transform, a 2D tensor with shape
    /// (samples x features).
    /// </param>
    /// <returns>
    /// A sequence of the transformed data, 2D tensors with shape (samples x components).
    /// </returns>
    public IObservable<Tensor> Process<T>(IObservable<Tuple<Tensor, T>> source) where T : PcaBaseModel
    {
        return source.Select(value =>
        {
            return TransformData(value.Item2, value.Item1);
        });
    }
}
