using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;

namespace Bonsai.ML.HiddenMarkovModels
{
    /// <summary>
    /// Represents the probabilities of being in each state of a Hidden Markov Model (HMM) given the observation.
    /// </summary>
    [Combinator]
    [Description("The probability of being in each state of a Hidden Markov Model (HMM) given the observation.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class StateProbability
    {
        /// <summary>
        /// The probability of being in each state given the observation.
        /// </summary>
        [XmlIgnore]
        [Description("The probability of being in each state given the observation.")]
        public double[] Probabilities { get; private set; }

        /// <summary>
        /// The state with the highest probability.
        /// </summary>
        [XmlIgnore]
        [Description("The state with the highest probability.")]
        public int HighestProbableState => Array.IndexOf(Probabilities, Probabilities.Max());

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="StateProbability"/> objects by accessing the `state_probabilities` attribute of the <see cref="PyObject"/>.
        /// </summary>
        public IObservable<StateProbability> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var probabilitiesPyObj = (double[])pyObject.GetArrayAttr("state_probabilities");

                return new StateProbability
                {
                    Probabilities = probabilitiesPyObj
                };
            });
        }
    }
}