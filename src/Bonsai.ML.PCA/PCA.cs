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
    public class PCA : PCABaseModel
    {
        public Tensor Covariance { get; private set; } = empty(0);
        public Tensor EigenValues { get; private set; } = empty(0);
        public Tensor EigenVectors { get; private set; } = empty(0);
        public Tensor Components { get; private set; } = empty(0);
        private bool _isFitted = false;

        public PCA(int numComponents,
            Device? device = null,
            ScalarType? scalarType = ScalarType.Float32)
            : base(numComponents,
                device,
                scalarType)
        { }

        public override void Fit(Tensor data)
        {
            if (data.NumberOfElements == 0 || data.dim() < 2)
            {
                throw new ArgumentException("Data must be a non-empty 2D tensor with shape (samples x features).", nameof(data));
            }

            var n = data.size(0);
            var d = data.size(1);

            if (NumComponents > d)
            {
                throw new ArgumentException("Number of components cannot be greater than the number of features.", nameof(data));
            }

            Covariance = cov(data);
            var eigen = eigh(Covariance);
            var sortedIndices = argsort(eigen.Item1, dim: -1, descending: true);
            EigenValues = eigen.Item1[sortedIndices];
            EigenVectors = eigen.Item2.index_select(1, sortedIndices);
            Components = EigenVectors.slice(1, 0, NumComponents, 1);
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

            var X = data.T;
            var mean = X.mean([0], keepdim: true); // 1 x d
            var Xc = X - mean;
            return Xc.matmul(Components); // n x q
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

            var X = data.T;
            var mean = X.mean([0], keepdim: true); // 1 x d
            var Xc = X - mean;
            var reconstructed = Transform(Xc);
            return reconstructed.matmul(Components.T) + mean.T;
        }
    }
}
