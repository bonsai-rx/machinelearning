using System;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.PCA;

internal static class Utils
{
    internal static Tensor InvertSPD(
        Tensor spdMatrix,
        Tensor rhs,
        double regularization = 1e-6,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        var diagShape = spdMatrix.size(-1);
        Tensor L;
        try
        {
            L = linalg.cholesky(spdMatrix);
        }
        catch (Exception)
        {
            var regularizer = eye(diagShape, device: device, dtype: scalarType) * regularization;
            L = linalg.cholesky(spdMatrix + regularizer);
        }
        return cholesky_solve(rhs, L);
    }
}