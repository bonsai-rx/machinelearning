using System;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// A standard Principal Component Analysis (PCA) model.
/// </summary>
public class Pca : PcaBaseModel
{
    /// <summary>
    /// Gets the mean of the fitted data.
    /// </summary>
    [XmlIgnore]
    public Tensor Mean { get; private set; } = empty(0);

    /// <summary>
    /// Gets the singular values of the fitted data.
    /// </summary>
    [XmlIgnore]
    public Tensor SingularValues { get; private set; } = empty(0);

    /// <inheritdoc/>
    protected override void FitCore(Tensor data)
    {
        if (NumComponents > data.size(0))
            throw new ArgumentException($"Number of components cannot be greater than the number of samples. Number of components: {NumComponents}, number of samples: {data.size(0)}.", nameof(data));

        var mean = data.mean([0], keepdim: true);
        var (_, singularValues, vh) = svd(data - mean, fullMatrices: false);

        Mean = mean.DetachFromDisposeScope();
        Components = vh.slice(0, 0, NumComponents, 1).T.contiguous().DetachFromDisposeScope();
        SingularValues = singularValues.slice(0, 0, NumComponents, 1).clone().DetachFromDisposeScope();
    }

    /// <inheritdoc/>
    protected override Tensor TransformCore(Tensor data)
    {
        return (data - Mean).matmul(Components);
    }

    /// <inheritdoc/>
    protected override Tensor ReconstructCore(Tensor data)
    {
        return data.matmul(Components.T) + Mean;
    }

    /// <inheritdoc/>
    public override void Dispose()
    {
        base.Dispose();
        Mean.Dispose();
        Mean = empty(0);
        SingularValues.Dispose();
        SingularValues = empty(0);
    }
}
