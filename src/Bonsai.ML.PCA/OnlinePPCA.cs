using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.PCA
{
    public class OnlinePPCA : PCABaseModel
    {
        public double? Rho { get; private set; }
        public double? Kappa { get; private set; }
        public double Variance => _sigma2.to_type(ScalarType.Float64).item<double>();
        public int ReorthogonalizePeriod { get; private set; }
        public int? TimeOffset { get; private set; }
        public Tensor Components => _W;
        public Generator Generator { get; private set; }

        private Tensor _mu;
        private Tensor _W;
        private Tensor _Iq;
        private Tensor _mx; // E[x]
        private Tensor _Cxz; // E[xz^T]
        private Tensor _mz; // E[z]
        private Tensor _Czz; // E[zz^T]
        private Tensor _sxx; // E[||x||^2]
        private Tensor _sigma2; // Variance

        private bool _initializedParameters = false;
        private readonly Func<double> UpdateSchedule;
        private int _stepCount = 0;
        private readonly bool _reorthogonalize = false;

        public OnlinePPCA(int numComponents,
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

            if (kappa.HasValue)
            {
                if (timeOffset is null or <= 0)
                {
                    throw new ArgumentException("Time offset must be a positive integer.", nameof(timeOffset));
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
                    _W = (orthonormalBases * _sigma2).MoveToOuterDisposeScope(); // d x q
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
                var M = _W.T.matmul(_W) + cov;
                var MInv = Utils.InvertSPD(M, _Iq);

                var XcW = Xc.matmul(_W);
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
                _W = WNew.index_select(1, indices).MoveToOuterDisposeScope();
                _Cxz = _Cxz.index_select(1, indices).MoveToOuterDisposeScope();
                _mz = _mz.index_select(0, indices).MoveToOuterDisposeScope();
                _Czz = _Czz.index_select(0, indices).index_select(1, indices).MoveToOuterDisposeScope();

                Sxz = _Cxz - outer(_mu, _mz);
                Szz = _Czz;

                // Update variance
                _sigma2 = ((Sxx - 2 * trace(_W.T.matmul(Sxz)) + trace(_W.T.matmul(_W).matmul(Szz))) / (double)d)
                    .clamp_min(0.0)
                    .MoveToOuterDisposeScope();
            }
        }

        public override Tensor Transform(Tensor data)
        {
            if (data.NumberOfElements == 0 || data.dim() < 2)
            {
                throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));
            }

            var Xt = data.T; // n x d
            var Xc = Xt - _mu; // n x d
            var M = _W.T.matmul(_W) + _Iq * _sigma2; // q x q
            var XcW = Xc.matmul(_W);
            return Utils.InvertSPD(M, XcW.T).T; // n x q
        }
        
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
            var M = _W.T.matmul(_W) + _Iq * _sigma2; // q x q
            var XcW = Xc.matmul(_W);
            var EzT = Utils.InvertSPD(M, XcW.T);
            var Ez = EzT.T;

            return Ez.matmul(_W.T) + _mu.T; // n x d
        }
    }
}
