using System;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Implements an online probabilistic PCA model using the stochastic online EM algorithm.
/// </summary>
public class OnlineProbabilisticPca : PcaBaseModel
{
    private Tensor _identityComponents = empty(0);
    private Tensor _mx = empty(0); // E[x]
    private Tensor _Cxz = empty(0); // E[xz^T]
    private Tensor _mz = empty(0); // E[z]
    private Tensor _Czz = empty(0); // E[zz^T]
    private Tensor _sxx = empty(0); // E[||x||^2]
    private readonly Func<double> UpdateSchedule;
    private int _stepCount = 0;
    private readonly bool _reorthogonalize = false;

    /// <summary>
    /// Rho is a constant learning rate parameter.
    /// </summary>
    /// <remarks>
    /// Rho must be in the range (0, 1). Only one of Rho or Kappa should be specified.
    /// </remarks>
    public double? Rho { get; private set; }

    /// <summary>
    /// Kappa is the exponent in the learning rate schedule.
    /// </summary>
    /// <remarks>
    /// Kappa must be in the range (0.5, 1]. Only one
    /// of Rho or Kappa should be specified.
    /// </remarks>
    public double? Kappa { get; private set; }

    /// <summary>
    /// Gets the mean of the fitted data.
    /// </summary>
    public Tensor Means { get; private set; } = empty(0);

    /// <summary>
    /// Gets the variance of the isotropic Gaussian noise model.
    /// </summary>
    public double Variance { get; private set; }

    /// <summary>
    /// Gets the period for reorthogonalizing the principal components.
    /// </summary>
    /// <remarks>
    /// Represented as the number of update steps between reorthogonalization operations.
    /// If not specified, reorthogonalization is not performed.
    /// </remarks>
    public int ReorthogonalizePeriod { get; private set; }

    /// <summary>
    /// Gets the sample offset used in the learning rate schedule when Kappa is specified.
    /// </summary>
    public int SampleOffset { get; private set; }

    /// <inheritdoc/>
    public override Tensor Components { get; protected set; } = empty(0);

    /// <summary>
    /// Gets the random number generator used for initializing the model.
    /// </summary>
    public Generator? Generator { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="OnlineProbabilisticPca"/> class.
    /// </summary>
    /// <param name="numComponents"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <param name="initialVariance"></param>
    /// <param name="generator"></param>
    /// <param name="rho"></param>
    /// <param name="kappa"></param>
    /// <param name="sampleOffset"></param>
    /// <param name="reorthogonalizePeriod"></param>
    /// <exception cref="ArgumentException"></exception>
    public OnlineProbabilisticPca(int numComponents,
        Device? device = null,
        ScalarType? scalarType = null,
        double initialVariance = 1.0,
        Generator? generator = null,
        double? rho = 0.1,
        double? kappa = null,
        int? sampleOffset = null,
        int? reorthogonalizePeriod = null
        ) : base(numComponents,
            device,
            scalarType)
    {
        if (initialVariance <= 0)
        {
            throw new ArgumentException("Starting variance must be greater than or equal to zero.", nameof(initialVariance));
        }

        if (kappa.HasValue && rho.HasValue)
        {
            throw new ArgumentException("Only one of rho or kappa should be specified, not both.", nameof(rho));
        }

        if (!kappa.HasValue && !rho.HasValue)
        {
            throw new ArgumentException("Either rho or kappa must be specified.", nameof(rho));
        }

        if (rho.HasValue)
        {
            if (rho.Value <= 0 || rho.Value >= 1)
            {
                throw new ArgumentException("Rho must be in the range (0, 1).", nameof(rho));
            }

            UpdateSchedule = () => rho.Value;
        }
        else
        {
            sampleOffset ??= 0;
            if (sampleOffset < 0)
            {
                throw new ArgumentException("Sample offset must be a positive integer.", nameof(sampleOffset));
            }

            if (!kappa.HasValue)
            {
                throw new ArgumentException("Kappa must be specified when using a learning rate schedule.", nameof(kappa));
            }

            if (kappa <= 0.5 || kappa > 1)
            {
                throw new ArgumentException("Kappa must be in the range (0.5, 1].", nameof(kappa));
            }

            UpdateSchedule = () => Math.Pow(_stepCount + SampleOffset, -kappa.Value);
        }

        if (reorthogonalizePeriod.HasValue)
        {
            _reorthogonalize = true;
            ReorthogonalizePeriod = reorthogonalizePeriod.Value;
        }

        Generator = generator;
        Rho = rho;
        Kappa = kappa;
        SampleOffset = sampleOffset ?? 0;
        Variance = initialVariance;
    }

    /// <inheritdoc/>
    public override void Fit(Tensor data)
    {
        base.Fit(data);

        using (no_grad())
        using (NewDisposeScope())
        {

            _stepCount++;
            var rho = UpdateSchedule();

            // Initialize dimensions
            var numSamples = data.size(0);

            // Initialize parameters
            if (Means.numel() == 0)
            {
                Means = zeros(NumFeatures, device: Device, dtype: ScalarType).MoveToOuterDisposeScope();
                var weights = qr(randn(NumFeatures, NumComponents, generator: Generator, device: Device, dtype: ScalarType), mode: QRMode.Reduced).Q;
                Components = (weights * Variance).MoveToOuterDisposeScope();
                _identityComponents = eye(NumComponents, device: Device, dtype: ScalarType).MoveToOuterDisposeScope();
                _mx = zeros(NumFeatures, device: Device, dtype: ScalarType).MoveToOuterDisposeScope(); // d
                _Cxz = zeros(NumFeatures, NumComponents, device: Device, dtype: ScalarType).MoveToOuterDisposeScope(); // d x q
                _mz = zeros(NumComponents, device: Device, dtype: ScalarType).MoveToOuterDisposeScope(); // q
                _Czz = zeros(NumComponents, NumComponents, device: Device, dtype: ScalarType).MoveToOuterDisposeScope(); // q x q
                _sxx = zeros(1, device: Device, dtype: ScalarType).MoveToOuterDisposeScope(); // scalar
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

            if (_reorthogonalize &&
                _stepCount % ReorthogonalizePeriod == 0)
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
                .to_type(TorchSharp.torch.ScalarType.Float64)
                .item<double>();
        }

        IsFitted = true;
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
}
