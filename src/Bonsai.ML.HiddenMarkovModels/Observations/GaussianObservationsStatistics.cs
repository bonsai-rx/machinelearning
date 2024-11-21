using System.ComponentModel;
using Python.Runtime;
using System.Xml.Serialization;
using System;
using System.Reactive.Linq;
using Bonsai.ML.Python;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Represents an operator that will transform an observable sequence of
    /// <see cref="PyObject"/> into an observable sequence of <see cref="GaussianObservationsStatistics"/>.
    /// </summary>
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class GaussianObservationsStatistics
    {
        /// <summary>
        /// The means of the observations for each state.
        /// </summary>
        [Description("The means of the observations for each state.")]
        [XmlIgnore]
        public double[,] Means { get; set; }

        /// <summary>
        /// The standard deviations of the observations for each state.
        /// </summary>
        [Description("The standard deviations of the observations for each state.")]
        [XmlIgnore]
        public double[,] StdDevs { get; set; }

        /// <summary>
        /// The covariance matrices of the observations for each state.
        /// </summary>
        [Description("The covariance matrices of the observations for each state.")]
        [XmlIgnore]
        public double[,,] CovarianceMatrices { get; set; }

        /// <summary>
        /// The batch observations that the model has seen.
        /// </summary>
        [Description("The batch observations that the model has seen.")]
        [XmlIgnore]
        public double[,] BatchObservations { get; set; }

        /// <summary>
        /// The predicted state for each observation in the batch of observations.
        /// </summary>
        [Description("The predicted state for each observation in the batch of observations.")]
        [XmlIgnore]
        public long[] PredictedStates { get; set; }

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
                var predictedStatesPyObj = (long[])pyObject.GetArrayAttr("predicted_states");

                return new GaussianObservationsStatistics
                {
                    Means = meansPyObj,
                    StdDevs = stdDevsPyObj,
                    CovarianceMatrices = covarianceMatricesPyObj,
                    BatchObservations = batchObservationsPyObj,
                    PredictedStates = PredictedStates
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