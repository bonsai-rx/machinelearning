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
    private Tensor _logConst = log(2 * Math.PI);

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
    /// Gets the log likelihood of the fitted model.
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

    private void CheckParameters()
    {
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
    public override void Fit(Tensor data)
    {
        CheckParameters();
        if (NumFeatures < 0)
        {
            Variance = InitialVariance;
        }

        base.Fit(data);

        using (no_grad())
        using (NewDisposeScope())
        {
            var numSamples = data.size(0);

            // Initialize log likelihood
            LogLikelihood = ones(Iterations, device: Device, dtype: Type) * double.NegativeInfinity;

            var weights = randn(NumFeatures, NumComponents, generator: Generator, device: Device, dtype: Type);
            var identityComponents = eye(NumComponents, device: Device, dtype: Type);
            var identityFeatures = eye(NumFeatures, device: Device, dtype: Type);

            var mean = data.mean([0], keepdim: true);
            var dataCentered = data - mean;

            // Calculate the sample covariance
            var covarianceTerm = dataCentered.T.matmul(dataCentered);
            var sampleCov = covarianceTerm / numSamples;

            // Calculate term 1 for variance update
            var term1 = trace(covarianceTerm);

            if (Device is not null)
                _logConst = _logConst.to(Device);

            // Compute log likelihood constant
            var logLikelihoodConst = NumFeatures * _logConst;

            double diffWeights;
            double diffVariance;

            // Repeat until convergence
            for (int i = 0; i < Iterations; i++)
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
                diffWeights = linalg.norm(weightsNew - weights).to_type(ScalarType.Float64).item<double>();
                diffVariance = abs(varianceNew - Variance).to_type(ScalarType.Float64).item<double>();

                // Update loglikelihood, weights and variance
                LogLikelihood[i] = logLikelihood;
                weights = weightsNew;
                Variance = varianceNew.to_type(ScalarType.Float64).item<double>();

                // Check for convergence
                if (diffWeights < Tolerance && diffVariance < Tolerance)
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
        Variance = 0.0;
        LogLikelihood = Utils.DisposeAndReset(LogLikelihood);
    }
}
