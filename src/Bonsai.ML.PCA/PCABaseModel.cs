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
    public abstract class PCABaseModel : IPCABaseModel
    {
        public int NumComponents { get; private set; }

        public Device Device { get; private set; }
        public ScalarType ScalarType { get; private set; }

        public PCABaseModel(int numComponents,
            Device? device = null,
            ScalarType? scalarType = null)
        {
            if (numComponents <= 0)
            {
                throw new ArgumentException("Number of components must be greater than zero.", nameof(numComponents));
            }

            NumComponents = numComponents;
            Device = device ?? CPU;
            ScalarType = scalarType ?? ScalarType.Float32;
        }

        public abstract void Fit(Tensor data);
        public abstract Tensor Transform(Tensor data);
        public virtual Tensor FitAndTransform(Tensor data)
        {
            Fit(data);
            return Transform(data);
        }
    }
}
