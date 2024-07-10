using System;
using System.ComponentModel;
using System.Reactive.Linq;
using Python.Runtime;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Represents an operator that will transform an observable sequence of
    /// <see cref="PyObject"/> into an observable sequence of <see cref="GaussianObservationsStatistics"/>.
    /// </summary>
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class GetGaussianObservationsStatisticsFromHMM
    {
        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="GaussianObservationsStatistics"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        public IObservable<GaussianObservationsStatistics> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var observationsPyObj = pyObject.GetAttr("observations");
                var meansPyObj = (double[,])observationsPyObj.GetArrayAttr("mus");
                var covarianceMatricesPyObj = (double[,,])observationsPyObj.GetArrayAttr("Sigmas");
                var stdDevsPyObj = DiagonalSqrt(covarianceMatricesPyObj);
                var batchObservationsPyObj = (double[,])pyObject.GetArrayAttr("batch_observations");
                var inferredMostProbableStatesPyObj = (long[])pyObject.GetArrayAttr("inferred_most_probable_states");

                return new GaussianObservationsStatistics
                {
                    Means = meansPyObj,
                    StdDevs = stdDevsPyObj,
                    CovarianceMatrices = covarianceMatricesPyObj,
                    BatchObservations = batchObservationsPyObj,
                    InferredMostProbableStates = inferredMostProbableStatesPyObj
                };
            });
        }

        private static double[,] DiagonalSqrt(double[,,] matrix)
        {
            var states = matrix.GetLength(0);
            var dimensions = matrix.GetLength(1);
            var diagonal = new double[states, dimensions];

            for (int i = 0; i < states; i++)
            {
                for (int j = 0; j < dimensions; j++)
                {
                    diagonal[i, j] = Math.Sqrt(matrix[i, j, j]);
                }
            }

            return diagonal;
        }
    }
}