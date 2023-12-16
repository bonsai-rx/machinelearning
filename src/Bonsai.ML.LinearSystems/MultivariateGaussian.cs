using MathNet.Numerics.LinearAlgebra;

namespace Bonsai.ML.LinearSystems
{
    /// <summary>
    /// Represents the result of a bayesian linear regression process.
    /// </summary>
    public class MultivariateGaussian
    {
        /// <summary>
        /// The estimated mean.
        /// </summary>
        public Vector<double> Mean;

        /// <summary>
        /// The estimated variance.
        /// </summary>
        public Matrix<double> Covariance;
    }
}
