using System;
using System.ComponentModel;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// A streaming/online PCA model based on the Generalized Hebbian Algorithm (GHA).
/// </summary>
/// <remarks>
/// Each call to fit the model updates the components with one step of the algorithm. The step
/// is scaled by the total variance of the data seen so far, so that the learning rate does not
/// depend on the scale of the data.
/// </remarks>
public class OnlinePcaGha : PcaBaseModel
{
    private double _totalVariance;

    /// <summary>
    /// Gets the number of samples that have been used to fit the model.
    /// </summary>
    [XmlIgnore]
    public long SampleCount { get; private set; } = 0;

    /// <summary>
    /// Gets the mean of the fitted data.
    /// </summary>
    [XmlIgnore]
    public Tensor Mean { get; private set; } = empty(0);

    /// <summary>
    /// Gets or sets the learning rate, relative to the total variance of the data.
    /// </summary>
    [Category("ModelParameters")]
    [Description("The learning rate used when updating the components, relative to the total variance of the data.")]
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
    protected override void ValidateParameters()
    {
        base.ValidateParameters();

        if (LearningRate < 0)
            throw new InvalidOperationException("Learning rate must be greater than or equal to zero.");
    }

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

        // Update the running mean and total variance
        var batchMean = data.mean([0], keepdim: true);
        var mean = SampleCount == 0 ? batchMean : Mean + (batchMean - Mean) * fraction;
        var dataCentered = data - mean;
        var batchVariance = dataCentered.pow(2).sum().to_type(ScalarType.Float64).item<double>() / numSamples;
        _totalVariance += (batchVariance - _totalVariance) * fraction;

        var stepSize = _totalVariance > 0 ? LearningRate / (numSamples * _totalVariance) : 0;

        var projection = dataCentered.matmul(components);
        var hebbianTerm = dataCentered.T.matmul(projection);
        var crossTerm = projection.T.matmul(projection);
        var correlation = components.matmul(crossTerm.triu(0));
        var weights = components + (hebbianTerm - correlation) * stepSize;

        // Orthonormalize, keeping each component pointing the same way as before the update
        var (Q, R) = linalg.qr(weights, mode: linalg.QRMode.Reduced);
        var signs = where(R.diagonal() < 0, -ones_like(R.diagonal()), ones_like(R.diagonal()));

        Components = (Q * signs).DetachFromDisposeScope();
        Mean = mean.DetachFromDisposeScope();
        SampleCount = totalSamples;
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
        _totalVariance = 0;
    }
}
