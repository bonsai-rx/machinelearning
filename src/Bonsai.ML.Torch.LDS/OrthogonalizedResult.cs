using TorchSharp;

namespace Bonsai.ML.Torch.LDS;

public struct OrthogonalizedResult
{
    public torch.Tensor OrthogonalizedState;
    public torch.Tensor OrthogonalizedCovariance;

    public OrthogonalizedResult(
        torch.Tensor orthogonalizedState,
        torch.Tensor orthogonalizedCovariance)
    {
        OrthogonalizedState = orthogonalizedState;
        OrthogonalizedCovariance = orthogonalizedCovariance;
    }
}