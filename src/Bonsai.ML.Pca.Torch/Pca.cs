using System;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Represents a standard Principal Component Analysis (PCA) model.
/// </summary>
public class Pca(int numComponents,
    Device? device = null,
    ScalarType? scalarType = null
) : PcaBaseModel(numComponents,
        device,
        scalarType)
{
    /// <summary>
    /// Gets the mean of the fitted data.
    /// </summary>
    public Tensor Mean { get; private set; } = empty(0);

    /// <inheritdoc/>
    public override Tensor Components { get; protected set; } = empty(0);

    /// <summary>
    /// The singular values of the fitted data.
    /// </summary>
    public Tensor SingularValues { get; private set; } = empty(0);

    /// <inheritdoc/>
    public override void Fit(Tensor data)
    {
        base.Fit(data);

        using (no_grad())
        using (NewDisposeScope())
        {
            var mean = data.mean([0], keepdim: true);
            var dataCentered = data - mean;
            var (U, S, Vh) = svd(dataCentered, fullMatrices: false);
            var components = Vh.slice(0, 0, NumComponents, 1).T;
            var singularValues = S.slice(0, 0, NumComponents, 1);

            if (ScalarType is not null && ScalarType != data.dtype)
            {
                var scalarType = ScalarType.Value;
                mean = mean.to(scalarType);
                components = components.to(scalarType);
                singularValues = singularValues.to(scalarType);
            }

            Mean = mean.MoveToOuterDisposeScope();
            Components = components.MoveToOuterDisposeScope();
            SingularValues = singularValues.MoveToOuterDisposeScope();
        }

        IsFitted = true;
    }

    /// <inheritdoc/>
    public override Tensor Transform(Tensor data)
    {
        base.Transform(data);
        var dataCentered = data - Mean;
        return dataCentered.matmul(Components);
    }

    /// <inheritdoc/>
    public override Tensor Reconstruct(Tensor data)
    {
        base.Reconstruct(data);
        return data.matmul(Components.T) + Mean;
    }
}
