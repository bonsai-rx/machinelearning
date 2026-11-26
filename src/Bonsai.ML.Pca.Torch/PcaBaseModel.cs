using System;
using System.ComponentModel;
using System.Linq;
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
    public int NumFeatures { get; private set; } = -1;

    /// <summary>
    /// Gets the components of the fitted model, a 2D tensor with shape (features x components).
    /// </summary>
    [XmlIgnore]
    public Tensor Components { get; protected set; } = empty(0);

    /// <summary>
    /// Gets or sets the number of principal components kept by the model.
    /// </summary>
    [Category("ModelParameters")]
    [Description("The number of principal components kept by the model.")]
    public int NumComponents { get; set; } = 2;

    /// <summary>
    /// Gets or sets the device on which the model operates. If not specified, the model operates
    /// on the device of the data it is fitted to.
    /// </summary>
    [XmlIgnore]
    [Description("The device on which the model operates. If not specified, the model operates on the device of the data it is fitted to.")]
    public Device? Device { get; set; }

    /// <summary>
    /// Gets or sets the data type used by the model. Input data is converted to this type.
    /// </summary>
    [Description("The data type used by the model. Input data is converted to this type.")]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Gets a value indicating whether each call to <see cref="Fit"/> updates the fitted model
    /// instead of replacing it.
    /// </summary>
    protected virtual bool IsIncremental => false;

    /// <summary>
    /// Fits the PCA model to the given data.
    /// </summary>
    /// <param name="data">The data to fit, a 2D tensor with shape (samples x features).</param>
    public void Fit(Tensor data)
    {
        ValidateParameters();
        CheckDataCompatibility(data);

        var numFeatures = (int)data.size(1);
        if (NumComponents > numFeatures)
            throw new ArgumentException($"Number of components cannot be greater than the number of features. Number of components: {NumComponents}, number of features: {numFeatures}.", nameof(data));

        var update = IsIncremental && NumFeatures >= 0;
        if (update)
            CheckFittedState(numFeatures);

        using (no_grad())
        using (NewDisposeScope())
        {
            var device = Device ?? (update ? Components.device : data.device);
            FitCore(Utils.Convert(data, Type, device));
        }

        NumFeatures = numFeatures;
    }

    /// <summary>
    /// Transforms the input data using the PCA model.
    /// </summary>
    /// <param name="data">
    /// The data to transform, a 2D tensor with shape (samples x features).
    /// </param>
    /// <returns>The transformed data, a 2D tensor with shape (samples x components).</returns>
    public Tensor Transform(Tensor data)
    {
        CheckFitted();
        CheckDataCompatibility(data);

        if (data.size(1) != NumFeatures)
            throw new ArgumentException("The number of features in the data does not match the number of features in the fitted model.", nameof(data));

        using (no_grad())
        using (NewDisposeScope())
        {
            var input = Utils.Convert(data, Components.dtype, Components.device);
            return TransformCore(input).MoveToOuterDisposeScope();
        }
    }

    /// <summary>
    /// Fits the PCA model to the given data and transforms it.
    /// </summary>
    /// <param name="data">
    /// The data to fit and transform, a 2D tensor with shape (samples x features).
    /// </param>
    /// <returns>The transformed data, a 2D tensor with shape (samples x components).</returns>
    public Tensor FitAndTransform(Tensor data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <summary>
    /// Reconstructs data in the original feature space from transformed data.
    /// </summary>
    /// <param name="data">
    /// The transformed data, a 2D tensor with shape (samples x components).
    /// </param>
    /// <returns>The reconstructed data, a 2D tensor with shape (samples x features).</returns>
    public Tensor Reconstruct(Tensor data)
    {
        CheckFitted();
        CheckDataCompatibility(data);

        if (data.size(1) != Components.size(1))
            throw new ArgumentException("The number of features in the data does not match the number of components in the fitted model.", nameof(data));

        using (no_grad())
        using (NewDisposeScope())
        {
            var input = Utils.Convert(data, Components.dtype, Components.device);
            return ReconstructCore(input).MoveToOuterDisposeScope();
        }
    }

    /// <summary>
    /// Checks that the model parameters are valid before fitting.
    /// </summary>
    protected virtual void ValidateParameters()
    {
        if (NumComponents <= 0)
            throw new InvalidOperationException("Number of components must be greater than zero.");
    }

    /// <summary>
    /// When overridden in a derived class, fits the model to data which has been validated and
    /// converted to the data type and device of the model.
    /// </summary>
    /// <param name="data">The data to fit, a 2D tensor with shape (samples x features).</param>
    protected abstract void FitCore(Tensor data);

    /// <summary>
    /// When overridden in a derived class, transforms data which has been validated and
    /// converted to the data type and device of the model.
    /// </summary>
    /// <param name="data">
    /// The data to transform, a 2D tensor with shape (samples x features).
    /// </param>
    /// <returns>The transformed data, a 2D tensor with shape (samples x components).</returns>
    protected abstract Tensor TransformCore(Tensor data);

    /// <summary>
    /// When overridden in a derived class, reconstructs data in the original feature space from
    /// transformed data which has been validated and converted to the data type and device of
    /// the model.
    /// </summary>
    /// <param name="data">
    /// The transformed data, a 2D tensor with shape (samples x components).
    /// </param>
    /// <returns>The reconstructed data, a 2D tensor with shape (samples x features).</returns>
    protected abstract Tensor ReconstructCore(Tensor data);

    private void CheckFitted()
    {
        if (NumFeatures < 0)
            throw new InvalidOperationException("Model has not yet been fitted. You should call one of the Fit() or the FitAndTransform() methods first.");
    }

    private void CheckFittedState(int numFeatures)
    {
        if (numFeatures != NumFeatures)
            throw new ArgumentException($"The number of features in the data does not match the number of features in the fitted model. Number of features in the data: {numFeatures}, number of features in the model: {NumFeatures}.", "data");

        if (Components.size(1) != NumComponents)
            throw new InvalidOperationException("The number of components cannot be changed once the model has been fitted.");

        if (Components.dtype != Type)
            throw new InvalidOperationException("The data type cannot be changed once the model has been fitted.");

        if (Device is not null && !Utils.IsOnDevice(Components, Device))
            throw new InvalidOperationException("The device cannot be changed once the model has been fitted.");
    }

    private static void CheckDataCompatibility(Tensor data)
    {
        if (data.NumberOfElements == 0)
            throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));

        if (data.dim() != 2)
        {
            var shapeStr = string.Join(",", data.shape.Select(x => x.ToString()).ToArray());
            throw new ArgumentException($"Data must be a 2D tensor with shape (samples x features). Data shape: {shapeStr}.", nameof(data));
        }
    }

    /// <summary>
    /// Releases all resources used by the PCA model and returns it to the unfitted state.
    /// </summary>
    public virtual void Dispose()
    {
        Components.Dispose();
        Components = empty(0);
        NumFeatures = -1;
    }
}
