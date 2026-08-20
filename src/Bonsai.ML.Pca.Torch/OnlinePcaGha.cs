using System.ComponentModel;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// A streaming/online PCA model based on the Generalized Hebbian Algorithm (GHA).
/// </summary>
public class OnlinePcaGha : PcaBaseModel
{
    /// <summary>
    /// Gets the number of samples that have been used to fit the model.
    /// </summary>
    [XmlIgnore]
    public int SampleCount { get; private set; } = 0;

    /// <summary>
    /// Gets the mean of the fitted data.
    /// </summary>
    [XmlIgnore]
    public Tensor Mean { get; private set; } = empty(0);

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    [Category("ModelParameters")]
    [Description("The learning rate used when updating the components.")]
    public double LearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the random number generator used for initializing the model.
    /// </summary>
    [XmlIgnore]
    [Description("The random number generator used for initializing the model.")]
    public Generator? Generator { get; set; }

    /// <inheritdoc/>
    public override void Fit(Tensor data)
    {
        base.Fit(data);

        var numSamples = data.size(0);

        using (no_grad())
        using (NewDisposeScope())
        {
            // Initialize components randomly
            if (Components.numel() == 0)
                Components = randn([NumFeatures, NumComponents], dtype: Type, device: Device, generator: Generator);

            if (Mean.numel() == 0)
                Mean = data.mean([0], keepdim: true);
            else
            {
                Mean *= (double)SampleCount / (SampleCount + numSamples);
                Mean += data.mean([0], keepdim: true) * numSamples / (SampleCount + numSamples);
            }

            SampleCount += (int)numSamples;
            var dataCentered = data - Mean;

            var projection = dataCentered.matmul(Components);
            var hebbianTerm = dataCentered.T.matmul(projection);
            var crossTerm = projection.T.matmul(projection);
            var upperTriangular = crossTerm.triu(0);
            var correlation = Components.matmul(upperTriangular);
            var componentsUpdate = (hebbianTerm - correlation) * (LearningRate / numSamples);
            var weights = Components + componentsUpdate;
            var norms = weights.norm(dim: 0, keepdim: true, p: 2).clamp_min(1e-12);

            Components = linalg.qr(weights / norms, mode: linalg.QRMode.Reduced).Q.MoveToOuterDisposeScope();
            Mean = Mean.MoveToOuterDisposeScope();
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
        SampleCount = 0;
    }
}
