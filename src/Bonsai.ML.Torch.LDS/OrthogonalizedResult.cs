using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Represents the result of orthogonalizing the state and covariance estimates.
/// </summary>
/// <remarks>
/// Initializes a new instance of the <see cref="OrthogonalizedResult"/> struct.
/// </remarks>
/// <param name="orthogonalizedState"></param>
/// <param name="orthogonalizedCovariance"></param>
public struct OrthogonalizedResult(
    Tensor orthogonalizedState,
    Tensor orthogonalizedCovariance)
{
    /// <summary>
    /// The orthogonalized state estimate.
    /// </summary>
    public Tensor OrthogonalizedState = orthogonalizedState;

    /// <summary>
    /// The orthogonalized covariance estimate.
    /// </summary>
    public Tensor OrthogonalizedCovariance = orthogonalizedCovariance;
}