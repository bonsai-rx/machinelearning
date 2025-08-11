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
    public class PPCA : PCABaseModel
    {
        public double Variance { get; private set; }
        public Tensor LogLikelihood { get; private set; } = empty(0);
        public Tensor Components { get; private set; } = empty(0);
        public Generator Generator { get; private set; }
        private int _iterations;
        private double _tolerance;
        private bool _isFitted = false;

        public PPCA(int numComponents,
            Device? device = null,
            ScalarType? scalarType = ScalarType.Float32,
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
            Generator = generator ?? manual_seed(0);
            _iterations = iterations;
            _tolerance = tolerance;
        }

        public override void Fit(Tensor data)
        {
            if (data.NumberOfElements == 0 || data.dim() != 2)
            {
                throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));
            }

            var Xt = data.T; // n x d

            // Initialize variance
            var variance = Variance;

            // Initialize log likelihood
            LogLikelihood = ones(_iterations) * double.NegativeInfinity;

            // Initialize dimensions for components
            var q = NumComponents;
            var n = Xt.size(0);
            var d = Xt.size(1);

            if (q > d)
            {
                throw new ArgumentException("Number of components cannot be greater than the number of features.", nameof(data));
            }

            // Initialize W and I
            var W = randn(d, q, generator: Generator); // d x q
            var MI = eye(q); // q x q
            var CI = eye(d); // d x d

            // Calculate the sample mean
            var mean = Xt.mean([0], keepdim: true); // 1 x d

            // Center the data and transpose
            var X = Xt - mean; // n x d

            // Calculate the sample covariance
            var XTX = X.T.matmul(X); // d x d
            var sampleCov = XTX / n; // d x d

            // Calculate term 1 for variance update
            var term1 = trace(XTX);

            // Compute log likelihood constant
            var logLikelihoodConst = d * log(2 * Math.PI);

            double diffW;
            double diffVariance;

            // Repeat until convergence
            for (int i = 0; i < _iterations; i++)
            {
                using (var _ = NewDisposeScope())
                {
                    // E-step: Compute the posterior distribution of the latent variables
                    var M = W.T.matmul(W) + MI * variance; // q x q
                    var MInv = inv(M); // q x q
                    var mu = MInv.matmul(W.T).matmul(X.T).T; // n x q
                    var SSum = n * MInv * variance; // q x q
                    var cov = mu.T.matmul(mu) + SSum; // q x q

                    // M-step: Compute new W and new variance
                    var XMu = X.T.matmul(mu); // d x q
                    var WNew = XMu.matmul(inv(cov)); // d x q

                    var term2 = 2 * XMu.mul(WNew).sum();
                    var mumu = mu.T.matmul(mu);
                    var WNewWNew = WNew.T.matmul(WNew);
                    var term3 = trace(WNewWNew.matmul(mumu + SSum));
                    var varianceNew = (term1 - term2 + term3) / (n * d); // scalar

                    // Compute the log likelihood
                    var C = W.matmul(W.T) + CI * variance; // d x d
                    var CInv = inv(C); // d x d
                    var logLikelihood = -0.5 * n * (logLikelihoodConst + logdet(C) + trace(CInv.matmul(sampleCov))); // scalar

                    // Check for convergence
                    diffW = linalg.norm(WNew - W).to_type(ScalarType.Float64).cpu().ReadCpuDouble(0);
                    diffVariance = abs(varianceNew - variance).to_type(ScalarType.Float64).cpu().ReadCpuDouble(0);

                    // Update loglikelihood, W and variance
                    LogLikelihood[i] = logLikelihood.MoveToOuterDisposeScope();
                    W = WNew.MoveToOuterDisposeScope();
                    variance = varianceNew.to_type(ScalarType.Float64).cpu().ReadCpuDouble(0);
                }


                if (diffW < _tolerance && diffVariance < _tolerance)
                {
                    LogLikelihood = LogLikelihood.slice(0, 0, i + 1, 1);
                    break;
                }
            }

            // Finalize model parameters
            LogLikelihood = LogLikelihood.DetachFromDisposeScope();
            Components = W.DetachFromDisposeScope();
            Variance = variance;
            _isFitted = true;
        }

        public override Tensor Transform(Tensor data)
        {
            if (data.NumberOfElements == 0 || data.dim() < 2)
            {
                throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));
            }

            if (!_isFitted)
            {
                throw new InvalidOperationException("Model has not yet been fitted. You should call the Fit() or the FitAndTransform() methods first.");
            }

            var Xt = data.T;
            var mean = Xt.mean([ 0 ], keepdim: true); // 1 x d
            var X = Xt - mean; // n x d
            var W = Components; // d x q
            var M = W.T.matmul(W) + eye(NumComponents) * Variance; // q x q
            var MInv = inverse(M); // q x q
            return X.matmul(W).matmul(MInv); // n x q
        }
        
        public override Tensor Reconstruct(Tensor data)
        {
            if (data.NumberOfElements == 0 || data.dim() < 2)
            {
                throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));
            }

            if (!_isFitted)
            {
                throw new InvalidOperationException("Model has not yet been fitted. You should call the Fit() or the FitAndTransform() methods first.");
            }

            var Xt = data.T;
            var mean = Xt.mean([0], keepdim: true); // 1 x d
            var Xc = Xt - mean; // n x d
            var W = Components; // d x q
            return Xc.matmul(W).matmul(W.T) + mean.T; // n x d
        }
    }
}
