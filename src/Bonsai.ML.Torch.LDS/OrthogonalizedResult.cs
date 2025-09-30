using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Represents the result of orthogonalizing the mean and covariance estimates.
/// </summary>
/// <remarks>
/// Initializes a new instance of the <see cref="OrthogonalizedResult"/> struct.
/// </remarks>
/// <param name="orthogonalizedMean"></param>
/// <param name="orthogonalizedCovariance"></param>
public struct OrthogonalizedResult(
    Tensor orthogonalizedMean,
    Tensor orthogonalizedCovariance)
{
    /// <summary>
    /// The orthogonalized mean estimate.
    /// </summary>
    public Tensor OrthogonalizedMean = orthogonalizedMean;

    /// <summary>
    /// The orthogonalized covariance estimate.
    /// </summary>
    public Tensor OrthogonalizedCovariance = orthogonalizedCovariance;
}