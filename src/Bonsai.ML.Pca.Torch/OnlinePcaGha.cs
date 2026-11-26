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
    protected override bool IsIncremental => true;

    /// <inheritdoc/>
    protected override void FitCore(Tensor data)
    {
        var numSamples = data.size(0);
        var numFeatures = data.size(1);
        var totalSamples = SampleCount + numSamples;
        var fraction = (double)numSamples / totalSamples;

        // Initialize components randomly
        var components = SampleCount == 0
            ? randn(numFeatures, NumComponents, dtype: data.dtype, device: data.device, generator: Generator)
            : Components;

        // Update the running mean
        var batchMean = data.mean([0], keepdim: true);
        var mean = SampleCount == 0 ? batchMean : Mean + (batchMean - Mean) * fraction;
        var dataCentered = data - mean;

        var projection = dataCentered.matmul(components);
        var hebbianTerm = dataCentered.T.matmul(projection);
        var crossTerm = projection.T.matmul(projection);
        var correlation = components.matmul(crossTerm.triu(0));
        var weights = components + (hebbianTerm - correlation) * (LearningRate / numSamples);
        var norms = weights.norm(dim: 0, keepdim: true, p: 2).clamp_min(1e-12);

        Components = linalg.qr(weights / norms, mode: linalg.QRMode.Reduced).Q.DetachFromDisposeScope();
        Mean = mean.DetachFromDisposeScope();
        SampleCount = (int)totalSamples;
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
        SampleCount = 0;
    }
}
