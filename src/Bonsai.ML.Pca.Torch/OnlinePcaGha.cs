using System;
using System.ComponentModel;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Implements online PCA using the Generalized Hebbian Algorithm (GHA).
/// </summary>
/// <param name="numComponents"></param>
/// <param name="learningRate"></param>
/// <param name="device"></param>
/// <param name="scalarType"></param>
/// <param name="generator"></param>
public class OnlinePcaGha(
    int numComponents,
    double learningRate = 0.1,
    Device? device = null,
    ScalarType? scalarType = null,
    Generator? generator = null
) : PcaBaseModel(numComponents, device, scalarType)
{
    /// <summary>
    /// Gets the number of samples that have been used to fit the model.
    /// </summary>
    public int SampleCount { get; private set; } = 0;

    /// <summary>
    /// Gets the mean of the fitted data.
    /// </summary>
    public Tensor Mean { get; private set; } = empty(0);

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    public double LearningRate { get; set; } = learningRate;

    /// <inheritdoc/>
    public override Tensor Components { get; protected set; } = empty(0);

    /// <summary>
    /// Gets the random number generator used for initializing the model.
    /// </summary>
    public Generator? Generator { get; private set; } = generator;

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
                Components = randn([NumFeatures, NumComponents], dtype: ScalarType, device: Device, generator: Generator);

            if (Mean.numel() == 0)
                Mean = data.mean([0], keepdim: true);
            else
            {
                Mean *= SampleCount / (SampleCount + numSamples);
                Mean += data.mean([0], keepdim: true) * numSamples / (SampleCount + numSamples);
            }

            SampleCount += (int)numSamples;
            var dataCentered = data - Mean;

            var projection = dataCentered.matmul(Components);
            var hebbianTerm = dataCentered.T.matmul(projection);
            var crossTerm = projection.T.matmul(projection);
            var lowerTriangular = crossTerm.tril(0);
            var correlation = Components.matmul(lowerTriangular);
            var componentsUpdate = (hebbianTerm - correlation) * (LearningRate / numSamples);
            var weights = Components + componentsUpdate;
            var norms = weights.norm(dim: 0, keepdim: true, p: 2).clamp_min(1e-12);

            Components = linalg.qr(weights / norms, mode: linalg.QRMode.Reduced).Q.MoveToOuterDisposeScope();
            Mean = Mean.MoveToOuterDisposeScope();
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
