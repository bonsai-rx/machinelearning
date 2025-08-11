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
        public int? BurnInCount { get; private set; }
        public double Variance { get; private set; }
        public Tensor Components => _W;
        public Generator Generator { get; private set; }
        private Tensor _mu;
        private Tensor _W;
        private Tensor _Iq;
        private Tensor _Id;
        private Tensor _Sx;
        private double _Sxxtrace;
        private Tensor _Sxz;
        private Tensor _Szz;
        private bool _initializedParameters = false;
        private int _count = 0;
        private int _burnInCount = 0;
        private Func<double> UpdateRho;

        public OnlinePPCA(int numComponents,
            double initialVariance = 1.0,
            Generator? generator = null,
            double? rho = 0.1,
            double? kappa = null,
            int? burnInCount = null
            ) : base(numComponents)
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
                UpdateRho = () => rho.Value;
                if (rho <= 0 || rho >= 1)
                {
                    throw new ArgumentException("Rho must be in the range (0, 1).", nameof(rho));
                }
            }

            if (kappa.HasValue)
            {
                UpdateRho = () => Math.Pow(_count + 10, -kappa.Value);
                if (kappa <= 0.5 || kappa > 1)
                {
                    throw new ArgumentException("Kappa must be in the range (0.5, 1].", nameof(kappa));
                }
            }

            if (burnInCount.HasValue && burnInCount <= 0)
            {
                throw new ArgumentException("Warmup iterations must be greater than zero.", nameof(burnInCount));
            }


            Variance = initialVariance;
            Generator = generator ?? manual_seed(0);
            Rho = rho;
            Kappa = kappa;
            BurnInCount = burnInCount;
            _burnInCount = burnInCount ?? 0;
        }

        public override void Fit(Tensor data)
        {
            // throw new NotImplementedException();
            if (data.NumberOfElements == 0 || data.dim() != 2)
            {
                throw new ArgumentException("Input data must be a 2D tensor.");
            }

            var Xt = data.T; // n x d

            // Initialize dimensions
            var q = NumComponents;
            var n = Xt.size(0);
            var d = Xt.size(1);

            // Initialize parameters
            if (!_initializedParameters)
            {
                _mu = zeros(1, d); // 1 x d
                var wScale = Math.Sqrt(Variance / d);
                _W = randn(d, q, generator: Generator) * wScale; // d x q
                _Iq = eye(q); // q x q
                _Id = eye(d); // d x d
                _Sx = zeros(1, d); // d
                _Sxz = zeros(d, q); // d x q
                _Szz = zeros(q, q); // q x q
                _Sxxtrace = 0.0; // scalar

                _burnInCount = Math.Max(_burnInCount, (int)d);
                _initializedParameters = true;
            }

            // Update rho
            var rho = UpdateRho();

            using (var _ = NewDisposeScope())
            {
                // E-step using current parameters
                var M = _W.T.matmul(_W) + _Iq * Variance; // q x q
                M = 0.5 * (M + M.T); // Ensure symmetry
                var Xc = Xt - _mu; // n x d
                var Lm = linalg.cholesky(M); // q x q
                var XcW = Xc.matmul(_W); // n x q
                var Ez = cholesky_solve(XcW.T, Lm).T; // n x q
                var EzInv = cholesky_inverse(Lm); // q x q
                var Ezzmu = (Variance * EzInv) + Ez.T.matmul(Ez);

                // Batch statistics
                var Xmu = Xt.mean([0], keepdim: true); // 1 x d
                var Xzmu = Xc.T.matmul(Ez); // d x q
                var Xcmu = Xc.pow(2).sum(1).mean();

                // Update parameters with new statistics
                _Sx = (1 - rho) * _Sx + rho * Xmu;
                _Sxz = (1 - rho) * _Sxz + rho * Xzmu;

                var SzzNew = (1 - rho) * _Szz + rho * Ezzmu;
                _Szz = 0.5 * (SzzNew + SzzNew.T); // Ensure symmetry
                _Sxxtrace = (1 - rho) * _Sxxtrace + rho * Xcmu.to_type(ScalarType.Float64).cpu().ReadCpuDouble(0);

                _Sx = _Sx.MoveToOuterDisposeScope();
                _Sxz = _Sxz.MoveToOuterDisposeScope();
                _Szz = _Szz.MoveToOuterDisposeScope();
                _Iq.MoveToOuterDisposeScope();
                _Id.MoveToOuterDisposeScope();

                // During burn-in, we do not update W or variance
                if (_count <= _burnInCount)
                {
                    _count++;
                    return;
                }

                // M-step: Update W and variance
                _mu = _Sx.MoveToOuterDisposeScope();
                var Lzz = linalg.cholesky(_Szz); // q x q
                var WNew = cholesky_solve(_Sxz.T, Lzz).T; // d x q
                // _W = WNew.MoveToOuterDisposeScope();
                var WUpdated = (1 - rho) * _W + rho * WNew;
                _W = WUpdated.MoveToOuterDisposeScope();

                var trWSt = WNew.mul(_Sxz).sum().to_type(ScalarType.Float64).cpu().ReadCpuDouble(0);
                var newVar = (_Sxxtrace - trWSt) / d;
                Variance = !double.IsNaN(newVar) && newVar > 0 ? newVar : Variance;
            }
        }

        public override Tensor Transform(Tensor data)
        {
            if (data.NumberOfElements == 0 || data.dim() < 2)
            {
                throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));
            }

            var Xt = data.T; // n x d
            var X = Xt - _mu; // n x d
            var W = Components; // d x q
            var M = W.T.matmul(W) + eye(NumComponents) * Variance; // q x q
            var MInv = inverse(M); // q x q
            return X.matmul(W).matmul(MInv); // n x q
        }
    }
}
