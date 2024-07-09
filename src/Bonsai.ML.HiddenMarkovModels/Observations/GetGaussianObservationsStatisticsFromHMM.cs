using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class GetGaussianObservationsStatisticsFromHMM
    {
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