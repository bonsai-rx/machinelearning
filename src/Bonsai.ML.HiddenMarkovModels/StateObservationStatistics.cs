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
    public class StateObservationStatistics
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

        public IObservable<StateObservationStatistics> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var meansPyObj = (double[,])pyObject.GetArrayAttr("observation_means");
                var stdDevsPyObj = (double[,])pyObject.GetArrayAttr("observation_stds");

                return new StateObservationStatistics
                {
                    Means = meansPyObj,
                    StdDevs = stdDevsPyObj
                };
            });
        }
    }
}