using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Defines the interface for PCA models.
/// </summary>
public interface IPcaBaseModel
{
    /// <summary>
    /// Gets the principal components of the model.
    /// </summary>
    public abstract Tensor Components { get; }

    /// <summary>
    /// Gets the number of principal components kept by the model.
    /// </summary>
    public int NumComponents { get; }

    /// <summary>
    /// Gets the device on which the model operates.
    /// </summary>
    public Device Device { get; }

    /// <summary>
    /// Gets the data type used by the model.
    /// </summary>
    public ScalarType ScalarType { get; }

    /// <summary>
    /// Fits the PCA model to the given data.
    /// </summary>
    /// <remarks>
    /// The input data should be a 2D tensor with shape (samples x features).
    /// </remarks>
    /// <param name="data"></param>
    public abstract void Fit(Tensor data);

    /// <summary>
    /// Transforms the input data using the PCA model.
    /// </summary>
    /// <remarks>
    /// The input data should be a 2D tensor with shape (samples x features).
    /// </remarks>
    /// <param name="data"></param>
    /// <returns></returns>
    public abstract Tensor Transform(Tensor data);

    /// <summary>
    /// Fits the PCA model to the given data and transforms it.
    /// </summary>
    /// <remarks>
    /// The input data should be a 2D tensor with shape (samples x features).
    /// </remarks>
    /// <param name="data"></param>
    /// <returns></returns>
    public abstract Tensor FitAndTransform(Tensor data);

    /// <summary>
    /// Reconstructs the input data using the PCA model.
    /// </summary>
    /// <remarks>
    /// The input data should be a 2D tensor with shape (samples x features).
    /// </remarks>
    /// <param name="data"></param>
    /// <returns></returns>
    public abstract Tensor Reconstruct(Tensor data);
}
