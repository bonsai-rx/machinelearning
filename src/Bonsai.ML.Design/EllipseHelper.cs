using MathNet.Numerics.LinearAlgebra;
using System;

namespace Bonsai.ML.Design
{
    /// <summary>
    /// Provides helper methods to compute ellipse parameters from a covariance matrix.
    /// </summary>
    public static class EllipseHelper
    {
        /// <summary>
        /// Computes the ellipse parameters from the specified covariance matrix.
        /// </summary>
        /// <param name="xVar">The variance of the x axis.</param>
        /// <param name="yVar">The variance of the y axis.</param>
        /// <param name="xyCov">The covariance between the x and y axes.</param>
        public static EllipseParameters GetEllipseParameters(double xVar, double yVar, double xyCov)
        {
            var covariance = Matrix<double>.Build.DenseOfArray(new double[,] {
                {
                    xVar,
                    xyCov
                },
                {
                    xyCov,
                    yVar
                },
            });

            var evd = covariance.Evd();
            var evals = evd.EigenValues.Real();
            evals = evals.PointwiseAbsoluteMaximum(0);
            var evecs = evd.EigenVectors;

            double angle = Math.Atan2(evecs[1, 0], evecs[0, 0]);

            return new EllipseParameters
            {
                Angle = angle,
                MajorAxis = Math.Sqrt(evals[0]),
                MinorAxis = Math.Sqrt(evals[1]),
            };
        }
    }

    /// <summary>
    /// Represents the parameters of an ellipse.
    /// </summary>
    public class EllipseParameters
    {
        /// <summary>
        /// Gets or sets the angle of the ellipse.
        /// </summary>
        public double Angle { get; set; }

        /// <summary>
        /// Gets or sets the major axis of the ellipse.
        /// </summary>
        public double MajorAxis { get; set; }

        /// <summary>
        /// Gets or sets the minor axis of the ellipse.
        /// </summary>
        public double MinorAxis { get; set; }
    }
}
