using System;
using System.ComponentModel;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// A probabilistic PCA model.
/// </summary>
public class ProbabilisticPca : PcaBaseModel
{
    /// <summary>
    /// Gets the mean of the fitted data.
    /// </summary>
    [XmlIgnore]
    public Tensor Mean { get; private set; } = empty(0);

    /// <summary>
    /// Gets the variance of the isotropic Gaussian noise model.
    /// </summary>
    [XmlIgnore]
    public double Variance { get; private set; } = 0.0;

    /// <summary>
    /// Gets the log likelihood of the fitted model at each iteration.
    /// </summary>
    [XmlIgnore]
    public Tensor LogLikelihood { get; private set; } = empty(0);

    /// <summary>
    /// Gets or sets the initial variance of the isotropic Gaussian noise model.
    /// </summary>
    [Category("ModelParameters")]
    [Description("The initial variance of the isotropic Gaussian noise model.")]
    public double InitialVariance { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the maximum number of iterations used for fitting the model.
    /// </summary>
    [Category("ModelParameters")]
    [Description("The maximum number of iterations used for fitting the model.")]
    public int Iterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the tolerance for convergence when fitting the model.
    /// </summary>
    [Category("ModelParameters")]
    [Description("The tolerance for convergence when fitting the model.")]
    public double Tolerance { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the random number generator used for initializing the model.
    /// </summary>
    [XmlIgnore]
    [Description("The random number generator used for initializing the model.")]
    public Generator? Generator { get; set; }

    /// <inheritdoc/>
    protected override void ValidateParameters()
    {
        base.ValidateParameters();

        if (InitialVariance < 0)
        {
            throw new InvalidOperationException("Initial variance must be greater than or equal to zero.");
        }

        if (Iterations <= 0)
        {
            throw new InvalidOperationException("Number of iterations must be greater than zero.");
        }

        if (Tolerance <= 0)
        {
            throw new InvalidOperationException("Tolerance must be greater than zero.");
        }
    }

    /// <inheritdoc/>
    protected override void FitCore(Tensor data)
    {
        var numSamples = data.size(0);
        var numFeatures = data.size(1);
        var type = data.dtype;
        var device = data.device;

        var identityComponents = eye(NumComponents, dtype: type, device: device);
        var identityFeatures = eye(numFeatures, dtype: type, device: device);
        var logLikelihood = ones(Iterations, dtype: type, device: device) * double.NegativeInfinity;

        var mean = data.mean([0], keepdim: true);
        var dataCentered = data - mean;

        // Calculate the sample covariance
        var scatter = dataCentered.T.matmul(dataCentered);
        var sampleCovariance = scatter / numSamples;
        var totalScatter = trace(scatter);

        var logLikelihoodConstant = numFeatures * Math.Log(2 * Math.PI);

        var weights = randn(numFeatures, NumComponents, dtype: type, device: device, generator: Generator);
        var variance = NumFeatures < 0 ? InitialVariance : Variance;
        var iterations = 0;

        // Repeat until convergence
        for (int i = 0; i < Iterations; i++)
        {
            using var iteration = NewDisposeScope();

            // E-step: Compute the posterior distribution of the latent variables
            var M = weights.T.matmul(weights) + identityComponents * variance;
            var MInv = inv(M);
            var mu = dataCentered.matmul(weights).matmul(MInv);
            var cov = mu.T.matmul(mu) + numSamples * MInv * variance;

            // M-step: Compute new weights and new variance
            var dataMu = dataCentered.T.matmul(mu);
            var weightsNew = dataMu.matmul(inv(cov));

            var term2 = 2 * dataMu.mul(weightsNew).sum();
            var term3 = trace(weightsNew.T.matmul(weightsNew).matmul(cov));
            var varianceNew = Math.Max(0, ((totalScatter - term2 + term3) / (numSamples * numFeatures))
                .to_type(ScalarType.Float64)
                .item<double>());

            // Compute the log likelihood
            var modelCovariance = weightsNew.matmul(weightsNew.T) + identityFeatures * varianceNew;
            var fit = logdet(modelCovariance) + trace(inv(modelCovariance).matmul(sampleCovariance));
            logLikelihood[i] = (fit + logLikelihoodConstant) * (-0.5 * numSamples);

            // Compare previous and new parameters for convergence
            var diffWeights = linalg.norm(weightsNew - weights).to_type(ScalarType.Float64).item<double>();
            var diffVariance = Math.Abs(varianceNew - variance);

            // Update weights and variance
            weights.Dispose();
            weights = weightsNew.MoveToOuterDisposeScope();
            variance = varianceNew;
            iterations = i + 1;

            // Check for convergence
            if (diffWeights < Tolerance && diffVariance < Tolerance)
                break;
        }

        // Finalize model parameters
        Components = weights.DetachFromDisposeScope();
        Mean = mean.DetachFromDisposeScope();
        Variance = variance;
        LogLikelihood = logLikelihood.slice(0, 0, iterations, 1).clone().DetachFromDisposeScope();
    }

    /// <inheritdoc/>
    protected override Tensor TransformCore(Tensor data)
    {
        return Utils.PosteriorMean(data - Mean, Components, Variance);
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
        Variance = 0.0;
        LogLikelihood.Dispose();
        LogLikelihood = empty(0);
    }
}
