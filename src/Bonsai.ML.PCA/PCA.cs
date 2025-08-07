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

        public PCA(int numComponents) : base(numComponents) { }

        public override void Fit(Tensor data)
        {
            if (data.NumberOfElements == 0 || data.dim() < 2)
            {
                throw new ArgumentException("Data must be a non-empty 2D tensor.", nameof(data));
            }

            Covariance = cov(data);
            var eigen = eigh(Covariance);
            var sortedIndices = argsort(eigen.Item1, dim: -1, descending: true);
            EigenValues = eigen.Item1[sortedIndices];
            EigenVectors = eigen.Item2.index_select(1, sortedIndices);
            Components = EigenVectors.slice(1, 0, NumComponents, 1);
        }

        public override Tensor Transform(Tensor data)
        {
            if (data.NumberOfElements == 0 || data.dim() < 2)
            {
                throw new ArgumentException("Data must be a non-empty 2D tensor.", nameof(data));
            }

            if (Components.NumberOfElements == 0)
            {
                throw new InvalidOperationException("Model has not been fit to data. Call the Fit() method first.");
            }

            return data.T.matmul(Components);
        }
    }
}
