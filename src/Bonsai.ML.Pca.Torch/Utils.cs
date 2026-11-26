using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

internal static class Utils
{
    // Solves A X = B for a symmetric positive definite matrix A
    internal static Tensor SolveSpd(Tensor matrix, Tensor rhs, double regularization = 1e-6)
    {
        var (factor, info) = linalg.cholesky_ex(matrix);
        if (info.item<int>() != 0)
        {
            var regularizer = eye(matrix.size(-1), dtype: matrix.dtype, device: matrix.device) * regularization;
            factor = linalg.cholesky(matrix + regularizer);
        }
        return cholesky_solve(rhs, factor);
    }

    // Posterior mean of the latent variables, (X - mean) W M^-1 with M = W^T W + variance I
    internal static Tensor PosteriorMean(Tensor dataCentered, Tensor weights, double variance)
    {
        var gram = weights.T.matmul(weights);
        var m = gram + eye(gram.size(0), dtype: gram.dtype, device: gram.device) * variance;
        return SolveSpd(m, weights.T.matmul(dataCentered.T)).T;
    }

    // Least squares reconstruction from posterior means, Z M (W^T W)^-1 W^T
    internal static Tensor Project(Tensor latent, Tensor weights, double variance)
    {
        var gram = weights.T.matmul(weights);
        var m = gram + eye(gram.size(0), dtype: gram.dtype, device: gram.device) * variance;
        return weights.matmul(SolveSpd(gram, m.matmul(latent.T))).T;
    }

    internal static bool IsOnDevice(Tensor tensor, Device device)
    {
        return tensor.device_type == device.type
            && (device.index < 0 || tensor.device_index == device.index);
    }

    internal static Tensor Convert(Tensor data, ScalarType type, Device device)
    {
        if (data.dtype != type)
            data = data.to_type(type);
        if (!IsOnDevice(data, device))
            data = data.to(device);
        return data;
    }

    // Replaces a tensor owned by a model, releasing the previous one
    internal static void Assign(ref Tensor field, Tensor value)
    {
        if (ReferenceEquals(field, value))
            return;
        field.Dispose();
        field = value.DetachFromDisposeScope();
    }
}
