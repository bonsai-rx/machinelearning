using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Implements an online Probabilistic Principal Component Analysis (PPCA) model using stochastic online EM.
/// </summary>
public class OnlineProbabilisticPca : PcaBaseModel
{
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
    /// Gets the variance of the isotropic Gaussian noise model.
    /// </summary>
    public double Variance => _sigma2.to_type(ScalarType.Float64).item<double>();

    /// <summary>
    /// Gets the period for reorthogonalizing the principal components.
    /// </summary>
    /// <remarks>
    /// Represented as the number of update steps between reorthogonalization operations.
    /// If not specified, reorthogonalization is not performed.
    /// </remarks>
    public int ReorthogonalizePeriod { get; private set; }

    /// <summary>
    /// Gets the time offset used in the learning rate schedule when Kappa is specified.
    /// </summary>
    public int? TimeOffset { get; private set; }

    /// <inheritdoc/>
    public override Tensor Components { get; protected set; } = empty(0);

    /// <summary>
    /// Gets the random number generator used for initializing the model.
    /// </summary>
    public Generator Generator { get; private set; }

    private Tensor _mu = empty(0);
    private Tensor _Iq = empty(0);
    private Tensor _mx = empty(0); // E[x]
    private Tensor _Cxz = empty(0); // E[xz^T]
    private Tensor _mz = empty(0); // E[z]
    private Tensor _Czz = empty(0); // E[zz^T]
    private Tensor _sxx = empty(0); // E[||x||^2]
    private Tensor _sigma2; // Variance

    private bool _initializedParameters = false;
    private readonly Func<double> UpdateSchedule;
    private int _stepCount = 0;
    private readonly bool _reorthogonalize = false;

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
    /// <param name="timeOffset"></param>
    /// <param name="reorthogonalizePeriod"></param>
    /// <exception cref="ArgumentException"></exception>
    public OnlineProbabilisticPca(int numComponents,
        Device? device = null,
        ScalarType? scalarType = ScalarType.Float32,
        double initialVariance = 1.0,
        Generator? generator = null,
        double? rho = 0.1,
        double? kappa = null,
        int? timeOffset = 3000,
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
            UpdateSchedule = () => rho.Value;
            if (rho <= 0 || rho >= 1)
            {
                throw new ArgumentException("Rho must be in the range (0, 1).", nameof(rho));
            }
        }
        else
        {
            if (timeOffset is null or <= 0)
            {
                throw new ArgumentException("Time offset must be a positive integer.", nameof(timeOffset));
            }

            if (!kappa.HasValue)
            {
                throw new ArgumentException("Kappa must be specified when using a learning rate schedule.", nameof(kappa));
            }

            UpdateSchedule = () => Math.Pow(_stepCount + timeOffset.Value, -kappa.Value);
            if (kappa <= 0.5 || kappa > 1)
            {
                throw new ArgumentException("Kappa must be in the range (0.5, 1].", nameof(kappa));
            }
        }

        if (reorthogonalizePeriod.HasValue)
        {
            _reorthogonalize = true;
            ReorthogonalizePeriod = reorthogonalizePeriod.Value;
        }

        Generator = generator ?? manual_seed(0);
        Rho = rho;
        Kappa = kappa;
        TimeOffset = timeOffset;
        _sigma2 = initialVariance;
    }

    /// <inheritdoc/>
    public override void Fit(Tensor data)
    {
        // throw new NotImplementedException();
        if (data.NumberOfElements == 0 || data.dim() != 2)
        {
            throw new ArgumentException("Input data must be a 2D tensor.");
        }

        using (no_grad())
        using (NewDisposeScope())
        {

            _stepCount++;
            var rho = UpdateSchedule();

            var Xt = data.T; // n x d

            // Initialize dimensions
            var q = NumComponents;
            var n = Xt.size(0);
            var d = Xt.size(1);

            // Initialize parameters
            if (!_initializedParameters)
            {
                _mu = zeros(d, device: Device, dtype: ScalarType).MoveToOuterDisposeScope();
                var randW = randn(d, q, generator: Generator, device: Device, dtype: ScalarType);
                var orthonormalBases = linalg.qr(randW).Q;
                Components = (orthonormalBases * _sigma2).MoveToOuterDisposeScope(); // d x q
                _Iq = eye(q, device: Device, dtype: ScalarType).MoveToOuterDisposeScope(); // q x q

                _mx = zeros(d, device: Device, dtype: ScalarType).MoveToOuterDisposeScope(); // d
                _Cxz = zeros(d, q, device: Device, dtype: ScalarType).MoveToOuterDisposeScope(); // d x q
                _mz = zeros(q, device: Device, dtype: ScalarType).MoveToOuterDisposeScope(); // q
                _Czz = zeros(q, q, device: Device, dtype: ScalarType).MoveToOuterDisposeScope(); // q x q
                _sxx = zeros(1, device: Device, dtype: ScalarType).MoveToOuterDisposeScope(); // scalar

                _initializedParameters = true;
            }

            // Covariance matrix
            var cov = _Iq * _sigma2;

            // Center data using current mean
            var Xc = Xt - _mu;

            // E-step
            var M = Components.T.matmul(Components) + cov;
            var MInv = Utils.InvertSPD(M, _Iq);

            var XcW = Xc.matmul(Components);
            var EzT = Utils.InvertSPD(M, XcW.T);
            var Ez = EzT.T;

            // Update statistics
            var mx = Xt.mean([0]);
            var sxx = Xt.pow(2).sum(dim: 1).mean();
            var Cxz = Xt.T.matmul(Ez) / n;
            var mz = Ez.mean([0]);
            var Czz = EzT.matmul(Ez) / n + _sigma2 * MInv;

            // Update parameters
            var rhoFactor = 1 - rho;
            _mx = (rhoFactor * _mx + rho * mx).MoveToOuterDisposeScope();
            _Cxz = (rhoFactor * _Cxz + rho * Cxz).MoveToOuterDisposeScope();
            _mz = (rhoFactor * _mz + rho * mz).MoveToOuterDisposeScope();
            _sxx = (rhoFactor * _sxx + rho * sxx).MoveToOuterDisposeScope();
            _Czz = (rhoFactor * _Czz + rho * Czz).MoveToOuterDisposeScope();

            // Update mean
            _mu = _mx.MoveToOuterDisposeScope();

            // Centered statistics
            var Sxz = _Cxz - outer(_mu, _mz);
            var Szz = _Czz;
            var Sxx = _sxx - _mu.dot(_mu);

            // M-step
            var WNew = Utils.InvertSPD(Szz, Sxz.T).T;

            if (_reorthogonalize &&
                _stepCount % ReorthogonalizePeriod == 0)
            {
                var (U, S, Vh) = svd(WNew, fullMatrices: false);
                var R = Vh.T;
                WNew = U.matmul(diag(S));
                _Cxz = _Cxz.matmul(R.T);
                _Czz = R.matmul(_Czz).matmul(R.T);
                _mz = R.matmul(_mz);
            }

            // Reorder components based on the strength of the components
            var strength = sum(WNew * WNew, dim: 0);
            var indices = argsort(strength, descending: true);
            Components = WNew.index_select(1, indices).MoveToOuterDisposeScope();
            _Cxz = _Cxz.index_select(1, indices).MoveToOuterDisposeScope();
            _mz = _mz.index_select(0, indices).MoveToOuterDisposeScope();
            _Czz = _Czz.index_select(0, indices).index_select(1, indices).MoveToOuterDisposeScope();

            Sxz = _Cxz - outer(_mu, _mz);
            Szz = _Czz;

            // Update variance
            _sigma2 = ((Sxx - 2 * trace(Components.T.matmul(Sxz)) + trace(Components.T.matmul(Components).matmul(Szz))) / (double)d)
                .clamp_min(0.0)
                .MoveToOuterDisposeScope();
        }
    }

    /// <inheritdoc/>
    public override Tensor Transform(Tensor data)
    {
        if (data.NumberOfElements == 0 || data.dim() < 2)
        {
            throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));
        }

        var Xt = data.T; // n x d
        var Xc = Xt - _mu; // n x d
        var M = Components.T.matmul(Components) + _Iq * _sigma2; // q x q
        var XcW = Xc.matmul(Components);
        return Utils.InvertSPD(M, XcW.T).T; // n x q
    }

    /// <inheritdoc/>
    public override Tensor Reconstruct(Tensor data)
    {
        if (data.NumberOfElements == 0 || data.dim() < 2)
        {
            throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));
        }

        if (!_initializedParameters)
        {
            throw new InvalidOperationException("Model has not yet been fitted. You should call the Fit() or the FitAndTransform() methods first.");
        }

        var Xt = data.T; // n x d
        var Xc = Xt - _mu; // n x d
        var M = Components.T.matmul(Components) + _Iq * _sigma2; // q x q
        var XcW = Xc.matmul(Components);
        var EzT = Utils.InvertSPD(M, XcW.T);
        var Ez = EzT.T;

        return Ez.matmul(Components.T) + _mu.T; // n x d
    }
}
