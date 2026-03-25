using System;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Represents a probabilistic PCA model.
/// </summary>
public class ProbabilisticPca : PcaBaseModel
{
    private readonly int _iterations;
    private readonly double _tolerance;

    /// <summary>
    /// Gets the mean of the fitted data.
    /// </summary>
    public Tensor Mean { get; private set; } = empty(0);

    /// <summary>
    /// Gets the variance of the isotropic Gaussian noise model.
    /// </summary>
    public double Variance { get; private set; }

    /// <summary>
    /// Gets the log likelihood of the fitted model.
    /// </summary>
    public Tensor LogLikelihood { get; private set; } = empty(0);

    /// <inheritdoc/>
    public override Tensor Components { get; protected set; } = empty(0);

    /// <summary>
    /// Gets the random number generator used for initializing the model.
    /// </summary>
    public Generator? Generator { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ProbabilisticPca"/> class.
    /// </summary>
    /// <param name="numComponents"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <param name="initialVariance"></param>
    /// <param name="generator"></param>
    /// <param name="iterations"></param>
    /// <param name="tolerance"></param>
    /// <exception cref="ArgumentException"></exception>
    public ProbabilisticPca(int numComponents,
        Device? device = null,
        ScalarType? scalarType = null,
        double initialVariance = 1.0,
        Generator? generator = null,
        int iterations = 100,
        double tolerance = 1e-5
        ) : base(numComponents,
            device,
            scalarType)
    {
        if (initialVariance < 0)
        {
            throw new ArgumentException("Starting variance must be greater than or equal to zero.", nameof(initialVariance));
        }

        if (iterations <= 0)
        {
            throw new ArgumentException("Number of iterations must be greater than zero.", nameof(iterations));
        }

        if (tolerance <= 0)
        {
            throw new ArgumentException("Tolerance must be greater than zero.", nameof(tolerance));
        }

        Variance = initialVariance;
        Generator = generator;
        _iterations = iterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc/>
    public override void Fit(Tensor data)
    {
        base.Fit(data);

        using (no_grad())
        using (NewDisposeScope())
        {
            var numSamples = data.size(0);

            // Initialize log likelihood
            LogLikelihood = ones(_iterations, device: Device, dtype: ScalarType) * double.NegativeInfinity;

            var weights = randn(NumFeatures, NumComponents, generator: Generator, device: Device, dtype: ScalarType);
            var identityComponents = eye(NumComponents, device: Device, dtype: ScalarType);
            var identityFeatures = eye(NumFeatures, device: Device, dtype: ScalarType);

            var mean = data.mean([0], keepdim: true);
            var dataCentered = data - mean;

            // Calculate the sample covariance
            var covarianceTerm = dataCentered.T.matmul(dataCentered);
            var sampleCov = covarianceTerm / numSamples;

            // Calculate term 1 for variance update
            var term1 = trace(covarianceTerm);

            // Compute log likelihood constant
            var logLikelihoodConst = NumFeatures * log(2 * Math.PI).to(Device);

            double diffWeights;
            double diffVariance;

            // Repeat until convergence
            for (int i = 0; i < _iterations; i++)
            {
                // E-step: Compute the posterior distribution of the latent variables
                var M = weights.T.matmul(weights) + identityComponents * Variance;
                var MInv = inv(M);
                var mu = MInv.matmul(weights.T).matmul(dataCentered.T).T;
                var SSum = numSamples * MInv * Variance;
                var cov = mu.T.matmul(mu) + SSum;

                // M-step: Compute new weights and new variance
                var dataMu = dataCentered.T.matmul(mu);
                var weightsNew = dataMu.matmul(inv(cov));

                var term2 = 2 * dataMu.mul(weightsNew).sum();
                var mu2 = mu.T.matmul(mu);
                var weightsNew2 = weightsNew.T.matmul(weightsNew);
                var term3 = trace(weightsNew2.matmul(mu2 + SSum));
                var varianceNew = (term1 - term2 + term3) / (numSamples * NumFeatures);

                // Compute the log likelihood
                var logLikelihoodTerm = weightsNew.matmul(weightsNew.T) + eye(NumFeatures) * varianceNew;
                var logLikelihoodTermInv = inv(logLikelihoodTerm);
                var logLikelihood = -0.5 * numSamples * (logLikelihoodConst + logdet(logLikelihoodTerm) + trace(logLikelihoodTermInv.matmul(sampleCov)));

                // Compare previous and new parameters for convergence
                diffWeights = linalg.norm(weightsNew - weights).to_type(TorchSharp.torch.ScalarType.Float64).item<double>();
                diffVariance = abs(varianceNew - Variance).to_type(TorchSharp.torch.ScalarType.Float64).item<double>();

                // Update loglikelihood, weights and variance
                LogLikelihood[i] = logLikelihood;
                weights = weightsNew;
                Variance = varianceNew.to_type(TorchSharp.torch.ScalarType.Float64).item<double>();

                // Check for convergence
                if (diffWeights < _tolerance && diffVariance < _tolerance)
                {
                    LogLikelihood = LogLikelihood.slice(0, 0, i + 1, 1);
                    break;
                }
            }

            // Finalize model parameters
            LogLikelihood = LogLikelihood.MoveToOuterDisposeScope();
            Components = weights.MoveToOuterDisposeScope();
            Mean = mean.MoveToOuterDisposeScope();
        }

        IsFitted = true;
    }

    /// <inheritdoc/>
    public override Tensor Transform(Tensor data)
    {
        base.Transform(data);
        var dataCentered = data - Mean;
        var M = Components.T.matmul(Components) + eye(NumComponents) * Variance;
        var MInv = Utils.InvertSPD(M, eye(NumComponents));
        return dataCentered.matmul(Components).matmul(MInv);
    }

    /// <inheritdoc/>
    public override Tensor Reconstruct(Tensor data)
    {
        base.Reconstruct(data);
        return data.matmul(Components.T) + Mean;
    }
}
