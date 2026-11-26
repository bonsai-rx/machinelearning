using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Represents an operator that fits a PCA model to the input data.
/// </summary>
[Combinator]
[Description("Fits a PCA model to the input data.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class Fit
{
    private void FitModel(PcaBaseModel model, Tensor data)
    {
        model.Fit(data);
    }

    /// <summary>
    /// Fits a PCA model to the input data.
    /// </summary>
    /// <typeparam name="T">The type of the PCA model.</typeparam>
    /// <param name="source">
    /// A sequence of pairs of a PCA model and the data to fit, a 2D tensor with shape
    /// (samples x features).
    /// </param>
    /// <returns>
    /// A sequence of the input pairs, each emitted once the model has been fitted to the data.
    /// </returns>
    public IObservable<Tuple<T, Tensor>> Process<T>(IObservable<Tuple<T, Tensor>> source) where T : PcaBaseModel
    {
        return source.Do((value) =>
        {
            FitModel(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Fits a PCA model to the input data.
    /// </summary>
    /// <typeparam name="T">The type of the PCA model.</typeparam>
    /// <param name="source">
    /// A sequence of pairs of a PCA model and the data to fit, a 2D tensor with shape
    /// (samples x features).
    /// </param>
    /// <returns>
    /// A sequence of the input pairs, each emitted once the model has been fitted to the data.
    /// </returns>
    public IObservable<Tuple<Tensor, T>> Process<T>(IObservable<Tuple<Tensor, T>> source) where T : PcaBaseModel
    {
        return source.Do((value) =>
        {
            FitModel(value.Item2, value.Item1);
        });
    }
}
