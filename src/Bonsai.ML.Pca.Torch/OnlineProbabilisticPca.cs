using System;
using System.ComponentModel;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// An online probabilistic PCA model which uses a stochastic online EM algorithm.
/// </summary>
public class OnlineProbabilisticPca : PcaBaseModel
{
    private Tensor _identityComponents = empty(0);
    private Tensor _mx = empty(0); // E[x]
    private Tensor _Cxz = empty(0); // E[xz^T]
    private Tensor _mz = empty(0); // E[z]
    private Tensor _Czz = empty(0); // E[zz^T]
    private Tensor _sxx = empty(0); // E[||x||^2]
    private int _stepCount = 0;

    /// <summary>
    /// Gets the mean of the fitted data.
    /// </summary>
    [XmlIgnore]
    public Tensor Means { get; private set; } = empty(0);

    /// <summary>
    /// Gets the variance of the isotropic Gaussian noise model.
    /// </summary>
    [XmlIgnore]
    public double Variance { get; private set; } = 0.0;

    /// <summary>
    /// Gets or sets the constant learning rate parameter.
    /// </summary>
    /// <remarks>
    /// Rho must be in the range (0, 1). Only one of Rho or Kappa should be specified.
    /// </remarks>
    [Category("ModelParameters")]
    [Description("The constant learning rate parameter. Only one of Rho or Kappa should be specified.")]
    public double? Rho { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the exponent in the learning rate schedule.
    /// </summary>
    /// <remarks>
    /// Kappa must be in the range (0.5, 1]. Only one
    /// of Rho or Kappa should be specified.
    /// </remarks>
    [Category("ModelParameters")]
    [Description("The exponent in the learning rate schedule. Only one of Rho or Kappa should be specified.")]
    public double? Kappa { get; set; }


    /// <summary>
    /// Gets or sets the initial variance of the isotropic Gaussian noise model.
    /// </summary>
    [Category("ModelParameters")]
    [Description("The initial variance of the isotropic Gaussian noise model.")]
    public double InitialVariance { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the period for reorthogonalizing the principal components.
    /// </summary>
    /// <remarks>
    /// Represented as the number of update steps between reorthogonalization operations.
    /// If not specified, reorthogonalization is not performed.
    /// </remarks>
    [Category("ModelParameters")]
    [Description("The number of update steps between reorthogonalization operations. If not specified, reorthogonalization is not performed.")]
    public int? ReorthogonalizePeriod { get; set; }

    /// <summary>
    /// Gets or sets the sample offset used in the learning rate schedule when Kappa is specified.
    /// </summary>
    [Category("ModelParameters")]
    [Description("The sample offset used in the learning rate schedule when Kappa is specified. If not specified, the decaying learning rate starts from the first sample.")]
    public int? SampleOffset { get; set; }

    /// <summary>
    /// Gets or sets the random number generator used for initializing the model.
    /// </summary>
    [XmlIgnore]
    [Description("The random number generator used for initializing the model.")]
    public Generator? Generator { get; set; }

    private void CheckParameters()
    {
        if (InitialVariance <= 0)
        {
            throw new InvalidOperationException("Initial variance must be greater than zero.");
        }

        if (Kappa.HasValue && Rho.HasValue)
        {
            throw new InvalidOperationException("Only one of Rho or Kappa should be specified, not both.");
        }

        if (!Kappa.HasValue && !Rho.HasValue)
        {
            throw new InvalidOperationException("Either Rho or Kappa must be specified.");
        }

        if (Rho.HasValue && (Rho.Value <= 0 || Rho.Value >= 1))
        {
            throw new InvalidOperationException("Rho must be in the range (0, 1).");
        }

        if (Kappa.HasValue && (Kappa.Value <= 0.5 || Kappa.Value > 1))
        {
            throw new InvalidOperationException("Kappa must be in the range (0.5, 1].");
        }

        if (SampleOffset < 0)
        {
            throw new InvalidOperationException("Sample offset must be a positive integer.");
        }
    }

    private double GetUpdateRate()
    {
        return Rho ?? Math.Pow(_stepCount + (SampleOffset ?? 0), -Kappa!.Value);
    }

    /// <inheritdoc/>
    public override void Fit(Tensor data)
    {
        CheckParameters();
        base.Fit(data);

        using (no_grad())
        using (NewDisposeScope())
        {

            _stepCount++;
            var rho = GetUpdateRate();

            // Initialize dimensions
            var numSamples = data.size(0);

            // Initialize parameters
            if (Means.numel() == 0)
            {
                Variance = InitialVariance;
                Means = zeros(NumFeatures, device: Device, dtype: Type).MoveToOuterDisposeScope();
                var weights = qr(randn(NumFeatures, NumComponents, generator: Generator, device: Device, dtype: Type), mode: QRMode.Reduced).Q;
                Components = (weights * Variance).MoveToOuterDisposeScope();
                _identityComponents = eye(NumComponents, device: Device, dtype: Type).MoveToOuterDisposeScope();
                _mx = zeros(NumFeatures, device: Device, dtype: Type).MoveToOuterDisposeScope(); // d
                _Cxz = zeros(NumFeatures, NumComponents, device: Device, dtype: Type).MoveToOuterDisposeScope(); // d x q
                _mz = zeros(NumComponents, device: Device, dtype: Type).MoveToOuterDisposeScope(); // q
                _Czz = zeros(NumComponents, NumComponents, device: Device, dtype: Type).MoveToOuterDisposeScope(); // q x q
                _sxx = zeros(1, device: Device, dtype: Type).MoveToOuterDisposeScope(); // scalar
            }

            // Covariance matrix
            var cov = _identityComponents * Variance;

            // Center data using current mean
            var dataCentered = data - Means;

            // E-step
            var M = Components.T.matmul(Components) + cov;
            var MInv = Utils.InvertSPD(M, _identityComponents);
            var projection = dataCentered.matmul(Components);
            var EzT = Utils.InvertSPD(M, projection.T);
            var Ez = EzT.T;

            // Update statistics
            var mx = data.mean([0]);
            var sxx = data.pow(2).sum(dim: 1).mean();
            var Cxz = data.T.matmul(Ez) / numSamples;
            var mz = Ez.mean([0]);
            var Czz = EzT.matmul(Ez) / numSamples + Variance * MInv;

            // Update parameters
            var rhoFactor = 1 - rho;
            _mx = (rhoFactor * _mx + rho * mx).MoveToOuterDisposeScope();
            _Cxz = (rhoFactor * _Cxz + rho * Cxz).MoveToOuterDisposeScope();
            _mz = (rhoFactor * _mz + rho * mz).MoveToOuterDisposeScope();
            _sxx = (rhoFactor * _sxx + rho * sxx).MoveToOuterDisposeScope();
            _Czz = (rhoFactor * _Czz + rho * Czz).MoveToOuterDisposeScope();

            // Update mean
            Means = _mx.MoveToOuterDisposeScope();

            // Centered statistics
            var Sxz = _Cxz - outer(Means, _mz);
            var Szz = _Czz;
            var Sxx = _sxx - Means.dot(Means);

            // M-step
            var weightsUpdated = Utils.InvertSPD(Szz, Sxz.T).T;

            if (ReorthogonalizePeriod.HasValue &&
                _stepCount % ReorthogonalizePeriod.Value == 0)
            {
                var (U, S, Vh) = svd(weightsUpdated, fullMatrices: false);
                var R = Vh.T;
                weightsUpdated = U.matmul(diag(S));
                _Cxz = _Cxz.matmul(R.T);
                _Czz = R.matmul(_Czz).matmul(R.T);
                _mz = R.matmul(_mz);
            }

            // Reorder components based on the strength of the components
            var strength = sum(weightsUpdated * weightsUpdated, dim: 0);
            var indices = argsort(strength, descending: true);
            Components = weightsUpdated.index_select(1, indices).MoveToOuterDisposeScope();
            _Cxz = _Cxz.index_select(1, indices).MoveToOuterDisposeScope();
            _mz = _mz.index_select(0, indices).MoveToOuterDisposeScope();
            _Czz = _Czz.index_select(0, indices).index_select(1, indices).MoveToOuterDisposeScope();

            Sxz = _Cxz - outer(Means, _mz);
            Szz = _Czz;

            // Update variance
            Variance = ((Sxx - 2 * trace(Components.T.matmul(Sxz)) + trace(Components.T.matmul(Components).matmul(Szz))) / (double)NumFeatures)
                .clamp_min(0.0)
                .to_type(ScalarType.Float64)
                .item<double>();
        }
    }

    /// <inheritdoc/>
    public override Tensor Transform(Tensor data)
    {
        base.Transform(data);
        var dataCentered = data - Means;
        var M = Components.T.matmul(Components) + _identityComponents * Variance;
        var projection = dataCentered.matmul(Components);
        return Utils.InvertSPD(M, projection.T).T;
    }

    /// <inheritdoc/>
    public override Tensor Reconstruct(Tensor data)
    {
        base.Reconstruct(data);
        return data.matmul(Components.T) + Means;
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

        Means = Utils.DisposeAndReset(Means);
        Variance = 0.0;

        _stepCount = 0;
        _identityComponents = Utils.DisposeAndReset(_identityComponents);
        _mx = Utils.DisposeAndReset(_mx);
        _Cxz = Utils.DisposeAndReset(_Cxz);
        _mz = Utils.DisposeAndReset(_mz);
        _Czz = Utils.DisposeAndReset(_Czz);
        _sxx = Utils.DisposeAndReset(_sxx);
    }
}
