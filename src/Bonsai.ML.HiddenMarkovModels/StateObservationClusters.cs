using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;

namespace Bonsai.ML.HiddenMarkovModels
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class StateObservationClusters
    {
        /// <summary>
        /// The means of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The means of the observations for each state.")]
        public double[,] Means { get; private set; }

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

        public IObservable<StateObservationClusters> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var meansPyObj = (double[,])pyObject.GetArrayAttr("observation_means");
                var covarianceMatricesPyObj = (double[,,])pyObject.GetArrayAttr("observation_covs");
                var batchObservationsPyObj = (double[,])pyObject.GetArrayAttr("batch_observations");
                var inferredMostProbableStatesPyObj = (long[])pyObject.GetArrayAttr("inferred_most_probable_states");

                return new StateObservationClusters
                {
                    Means = meansPyObj,
                    CovarianceMatrices = covarianceMatricesPyObj,
                    BatchObservations = batchObservationsPyObj,
                    InferredMostProbableStates = inferredMostProbableStatesPyObj
                };
            });
        }
    }
}