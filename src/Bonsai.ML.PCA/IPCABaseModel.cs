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
    public interface IPCABaseModel
    {
        public Device Device { get; }
        public ScalarType ScalarType { get; }
        public abstract void Fit(Tensor data);
        public abstract Tensor Transform(Tensor data);
        public abstract Tensor FitAndTransform(Tensor data);
        public abstract Tensor Reconstruct(Tensor data);
    }
}
