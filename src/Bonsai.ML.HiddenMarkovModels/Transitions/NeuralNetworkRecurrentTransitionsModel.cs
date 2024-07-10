using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using Newtonsoft.Json;
using System.Xml.Serialization;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// Represents an operator that is used to create and transform an observable sequence
    /// of <see cref="NeuralNetworkRecurrentTransitions"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of NeuralNetworkRecurrentTransitions objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class NeuralNetworkRecurrentTransitionsModel
    {
        /// <summary>
        /// The sizes of the hidden layers.
        /// </summary>
        [XmlIgnore]
        [Description("The sizes of the hidden layers.")]
        [JsonProperty]
        public int[] HiddenLayerSizes { get; set; } = [50];

        /// <summary>
        /// The Log Ps of the transitions.
        /// </summary>
        [XmlIgnore]
        [Description("The log Ps of the transitions.")]
        public double[,] LogPs { get; set; } = null;

        /// <summary>
        /// The weights.
        /// </summary>
        [XmlIgnore]
        [Description("The weights.")]
        public double[,,] Weights { get; set; } = null;

        /// <summary>
        /// The biases.
        /// </summary>
        [XmlIgnore]
        [Description("The biases.")]
        public double[,,] Biases { get; set; } = null;

        /// <summary>
        /// Returns an observable sequence of <see cref="NeuralNetworkRecurrentTransitions"/> objects.
        /// </summary>
        new public IObservable<NeuralNetworkRecurrentTransitions> Process()
        {
            return Observable.Return(
                new NeuralNetworkRecurrentTransitions(HiddenLayerSizes)
                {
                    Params = [LogPs, Weights, Biases]
                });
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="NeuralNetworkRecurrentTransitions"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        new public IObservable<NeuralNetworkRecurrentTransitions> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var logPsPyObj = (double[,])pyObject.GetArrayAttr("log_Ps");
                var weightsPyObj = (double[,])pyObject.GetArrayAttr("weights");
                var biasesPyObj = (double[,])pyObject.GetArrayAttr("biases");

                return new NeuralNetworkRecurrentTransitions()
                {
                    Params = [logPsPyObj, weightsPyObj, biasesPyObj]
                };
            });
        }
    }
}