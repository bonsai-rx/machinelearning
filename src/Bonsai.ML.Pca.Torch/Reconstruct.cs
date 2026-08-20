using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Represents an operator that reconstructs the input data from transformed data using a PCA model.
/// </summary>
[Combinator]
[Description("Reconstructs the input data from transformed data using a PCA model.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Reconstruct
{
    private static Tensor ReconstructData(PcaBaseModel model, Tensor data)
    {
        return model.Reconstruct(data);
    }

    /// <summary>
    /// Reconstructs the input data from transformed data using a PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process<T>(IObservable<Tuple<T, Tensor>> source) where T : PcaBaseModel
    {
        return source.Select(value =>
        {
            return ReconstructData(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Reconstructs the input from transformed data using a PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process<T>(IObservable<Tuple<Tensor, T>> source) where T : PcaBaseModel
    {
        return source.Select(value =>
        {
            return ReconstructData(value.Item2, value.Item1);
        });
    }
}
