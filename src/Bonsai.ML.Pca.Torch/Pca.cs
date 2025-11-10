using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Implements a standard Principal Component Analysis (PCA) model.
/// </summary>
public class Pca(int numComponents,
    Device? device = null,
    ScalarType? scalarType = ScalarType.Float32) : PcaBaseModel(numComponents,
        device,
        scalarType)
{
    /// <summary>
    /// Gets the covariance matrix of the fitted data.
    /// </summary>
    public Tensor Covariance { get; private set; } = empty(0);

    /// <summary>
    /// Gets the eigenvalues of the covariance matrix.
    /// </summary>
    public Tensor EigenValues { get; private set; } = empty(0);

    /// <summary>
    /// Gets the eigenvectors of the covariance matrix.
    /// </summary>
    public Tensor EigenVectors { get; private set; } = empty(0);

    /// <inheritdoc/>
    public override Tensor Components { get; protected set; } = empty(0);

    private bool _isFitted = false;

    /// <inheritdoc/>
    public override void Fit(Tensor data)
    {
        if (data.NumberOfElements == 0 || data.dim() < 2)
        {
            throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));
        }

        var d = data.size(1);

        if (NumComponents > d)
        {
            throw new ArgumentException("Number of components cannot be greater than the number of features.", nameof(data));
        }

        var Xt = data.T;
        var mean = Xt.mean([0], keepdim: true);
        var Xc = Xt - mean;
        Covariance = cov(Xc.T);
        var eigen = eigh(Covariance);
        var sortedIndices = argsort(eigen.Item1, dim: -1, descending: true);
        EigenValues = eigen.Item1[sortedIndices];
        EigenVectors = eigen.Item2.index_select(1, sortedIndices);
        Components = EigenVectors.slice(1, 0, NumComponents, 1);
        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Tensor Transform(Tensor data)
    {
        if (data.NumberOfElements == 0 || data.dim() < 2)
        {
            throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));
        }

        if (!_isFitted)
        {
            throw new InvalidOperationException("Model has not yet been fitted. You should call the Fit() or the FitAndTransform() methods first.");
        }

        var X = data.T;
        var mean = X.mean([0], keepdim: true); // 1 x d
        var Xc = X - mean;
        return Xc.matmul(Components); // n x q
    }

    /// <inheritdoc/>
    public override Tensor Reconstruct(Tensor data)
    {
        if (data.NumberOfElements == 0 || data.dim() < 2)
        {
            throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));
        }

        if (!_isFitted)
        {
            throw new InvalidOperationException("Model has not yet been fitted. You should call the Fit() or the FitAndTransform() methods first.");
        }

        var X = data.T;
        var mean = X.mean([0], keepdim: true); // 1 x d
        var Xc = X - mean;
        var reconstructed = Transform(Xc);
        return reconstructed.matmul(Components.T) + mean.T;
    }
}
