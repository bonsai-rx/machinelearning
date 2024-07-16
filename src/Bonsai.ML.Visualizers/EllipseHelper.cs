using MathNet.Numerics.LinearAlgebra;
using System;

namespace Bonsai.ML.Visualizers
{
    public static class EllipseHelper
    {
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

    public class EllipseParameters
    {
        public double Angle { get; set; }
        public double MajorAxis { get; set; }
        public double MinorAxis { get; set; }
    }
}
