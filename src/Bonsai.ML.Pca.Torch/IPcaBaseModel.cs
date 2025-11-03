using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

public interface IPcaBaseModel
{
    public Device Device { get; }
    public ScalarType ScalarType { get; }
    public abstract void Fit(Tensor data);
    public abstract Tensor Transform(Tensor data);
    public abstract Tensor FitAndTransform(Tensor data);
    public abstract Tensor Reconstruct(Tensor data);
}
