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
    public class GaussianObservationsStatistics
    {
        /// <summary>
        /// The means of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The means of the observations for each state.")]
        public double[,] Means { get; private set; }

        /// <summary>
        /// The standard deviations of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The standard deviations of the observations for each state.")]
        public double[,] StdDevs { get; private set; }

        /// <summary>
        /// The covariance matrices of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The covariance matrices of the observations for each state.")]
        public double[,,] CovarianceMatrices { get; private set; }

        /// <summary>
        /// The batch observations that the model has seen.
        /// </summary>
        [XmlIgnore]
        [Description("The batch observations that the model has seen.")]
        public double[,] BatchObservations { get; private set; }

        /// <summary>
        /// The sequence of inferred most probable states.
        /// </summary>
        [XmlIgnore]
        [Description("The sequence of inferred most probable states.")]
        public long[] InferredMostProbableStates { get; private set; }

        public IObservable<GaussianObservationsStatistics> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var observationsPyObj = pyObject.GetAttr("observations");
                var meansPyObj = (double[,])observationsPyObj.GetArrayAttr("mus");
                var stdDevsPyObj = GetDiagonal((double[,,])observationsPyObj.GetArrayAttr("Sigmas"));
                var covarianceMatricesPyObj = (double[,,])observationsPyObj.GetArrayAttr("_sqrt_Sigmas");
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

        private static double[,] GetDiagonal(double[,,] matrix)
        {
            var states = matrix.GetLength(0);
            var dimensions = matrix.GetLength(1);
            var diagonal = new double[states, dimensions];

            for (int i = 0; i < states; i++)
            {
                for (int j = 0; j < dimensions; j++)
                {
                    diagonal[i, j] = matrix[i, j, j];
                }
            }

            return diagonal;
        }
    }
}