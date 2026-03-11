using System;
using System.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Provides an abstract base class for PCA models.
/// </summary>
public abstract class PcaBaseModel : IPcaBaseModel
{
    /// <inheritdoc/>
    public bool IsFitted { get; protected set; }

    /// <inheritdoc/>
    public int NumFeatures { get; protected set; } = -1;

    /// <inheritdoc/>
    public abstract Tensor Components { get; protected set; }

    /// <inheritdoc/>
    public int NumComponents { get; private set; }

    /// <inheritdoc/>
    public Device Device { get; private set; }

    /// <inheritdoc/>
    public ScalarType? ScalarType { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="PcaBaseModel"/> class.
    /// </summary>
    /// <param name="numComponents"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <exception cref="ArgumentException"></exception>
    public PcaBaseModel(int numComponents,
        Device? device = null,
        ScalarType? scalarType = null)
    {
        if (numComponents <= 0)
        {
            throw new ArgumentException("Number of components must be greater than zero.", nameof(numComponents));
        }

        NumComponents = numComponents;
        Device = device ?? CPU;
        ScalarType = scalarType;
    }

    /// <inheritdoc/>
    public virtual void Fit(Tensor data)
    {
        CheckDataCompatibility(data);

        var d = data.size(1);

        if (NumComponents > d)
            throw new ArgumentException($"Number of components cannot be greater than the number of features. Number of components: {NumComponents}, number of features: {d}.", nameof(data));

        NumFeatures = (int)d;
    }

    /// <inheritdoc/>
    public virtual Tensor Transform(Tensor data)
    {
        CheckFitted();
        CheckDataCompatibility(data);
        CheckDataFeatures(data);
        return data;
    }

    /// <inheritdoc/>
    public virtual Tensor FitAndTransform(Tensor data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc/>
    public virtual Tensor Reconstruct(Tensor data)
    {
        CheckFitted();
        CheckDataCompatibility(data);
        CheckDataFeatures(data);
        return data;
    }

    private void CheckFitted()
    {
        if (!IsFitted)
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
}
