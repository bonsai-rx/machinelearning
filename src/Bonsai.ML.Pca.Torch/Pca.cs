using System.ComponentModel;
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

            if (Type != data.dtype)
            {
                mean = mean.to(Type);
                components = components.to(Type);
                singularValues = singularValues.to(Type);
            }

            Mean = mean.MoveToOuterDisposeScope();
            Components = components.MoveToOuterDisposeScope();
            SingularValues = singularValues.MoveToOuterDisposeScope();
        }
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

    /// <inheritdoc/>
    public override Tensor FitAndTransform(Tensor data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc/>
    public override void Dispose()
    {
        base.Dispose();
        Mean = Utils.DisposeAndReset(Mean);
        SingularValues = Utils.DisposeAndReset(SingularValues);
    }
}
