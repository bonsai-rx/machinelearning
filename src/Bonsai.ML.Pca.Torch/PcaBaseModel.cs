using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Provides an abstract base class for PCA models.
/// </summary>
public abstract class PcaBaseModel : IDisposable
{
    /// <summary>
    /// Gets the number of features in the fitted data.
    /// </summary>
    [XmlIgnore]
    public int NumFeatures { get; protected set; } = -1;

    /// <inheritdoc/>
    [XmlIgnore]
    public Tensor Components { get; protected set; } = empty(0);

    /// <summary>
    /// Gets or sets the number of principal components kept by the model.
    /// </summary>
    [Category("ModelParameters")]
    [Description("The number of principal components kept by the model.")]
    public int NumComponents { get; set; } = 2;

    /// <summary>
    /// Gets or sets the device on which the model operates.
    /// </summary>
    [XmlIgnore]
    [Description("The device on which the model operates.")]
    public Device? Device { get; set; }

    /// <summary>
    /// Gets or sets the data type used by the model.
    /// </summary>
    [Description("The data type used by the model.")]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Fits the PCA model to the given data.
    /// </summary>
    /// <remarks>
    /// The input data should be a 2D tensor with shape (samples x features).
    /// </remarks>
    /// <param name="data"></param>
    public virtual void Fit(Tensor data)
    {

        if (NumComponents <= 0)
            throw new InvalidOperationException("Number of components must be greater than zero.");

        CheckDataCompatibility(data);

        var d = data.size(1);

        if (NumComponents > d)
            throw new ArgumentException($"Number of components cannot be greater than the number of features. Number of components: {NumComponents}, number of features: {d}.", nameof(data));

        NumFeatures = (int)d;
    }

    /// <summary>
    /// Transforms the input data using the PCA model.
    /// </summary>
    /// <remarks>
    /// The input data should be a 2D tensor with shape (samples x features).
    /// </remarks>
    /// <param name="data"></param>
    /// <returns></returns>
    public virtual Tensor Transform(Tensor data)
    {
        CheckFitted();
        CheckDataCompatibility(data);
        CheckDataFeatures(data);
        return data;
    }

    /// <summary>
    /// Fits the PCA model to the given data and transforms it.
    /// </summary>
    /// <remarks>
    /// The input data should be a 2D tensor with shape (samples x features).
    /// </remarks>
    /// <param name="data"></param>
    /// <returns></returns>
    public virtual Tensor FitAndTransform(Tensor data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <summary>
    /// Reconstructs the input data using the PCA model.
    /// </summary>
    /// <remarks>
    /// The input data should be a 2D tensor with shape (samples x features).
    /// </remarks>
    /// <param name="data"></param>
    /// <returns></returns>
    public virtual Tensor Reconstruct(Tensor data)
    {
        CheckFitted();
        CheckDataCompatibility(data);
        CheckDataComponents(data);
        return data;
    }

    private void CheckFitted()
    {
        if (NumFeatures < 0)
            throw new InvalidOperationException("Model has not yet been fitted. You should call one of the Fit() or the FitAndTransform() methods first.");
    }

    private void CheckDataCompatibility(Tensor data)
    {
        if (data.NumberOfElements == 0)
            throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));

        if (data.dim() != 2)
        {
            var shapeStr = string.Join(",", data.shape.Select(x => x.ToString()).ToArray());
            throw new ArgumentException($"Data must be a 2D tensor with shape (samples x features). Data shape: {shapeStr}.", nameof(data));
        }
    }

    private void CheckDataFeatures(Tensor data)
    {
        var d = data.size(1);

        if (d != NumFeatures)
            throw new ArgumentException("The number of features in the data does not match the number of features in the fitted model.", nameof(data));
    }

    private void CheckDataComponents(Tensor data)
    {
        var d = data.size(1);

        if (d != NumComponents)
            throw new ArgumentException("The number of features in the data does not match the number of components in the fitted model.", nameof(data));
    }

    /// <summary>
    /// Releases all resources used by the PCA model.
    /// </summary>
    public virtual void Dispose()
    {
        Components = Utils.DisposeAndReset(Components);
        NumFeatures = -1;
    }
}
