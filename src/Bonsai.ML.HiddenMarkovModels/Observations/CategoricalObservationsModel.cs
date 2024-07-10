using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Represents an operator that is used to create and transform an observable sequence
    /// of <see cref="CategoricalObservations"/> objects.
    /// </summary>
    [Combinator]
    [Description("Creates an observable sequence of CategoricalObservations objects.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonObject(MemberSerialization.OptIn)]
    public class CategoricalObservationsModel
    {
        /// <summary>
        /// The number of categories in the observations.
        /// </summary>
        [Description("The number of categories in the observations.")]
        [JsonProperty]
        public int Categories { get; set; } = 2;

        /// <summary>
        /// The logit of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The logit of the observations for each state.")]
        public double[,,] Logits { get; set; } = null;

        /// <summary>
        /// Returns an observable sequence of <see cref="CategoricalObservations"/> objects.
        /// </summary>
        public IObservable<CategoricalObservations> Process()
        {
            return Observable.Return(
                new CategoricalObservations (Categories) {
                    Params = [ Logits ],
                });
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="CategoricalObservations"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        public IObservable<CategoricalObservations> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var categoriesPyObj = (int)pyObject.GetArrayAttr("C");
                var logitsPyObj = (double[,,])pyObject.GetArrayAttr("logits");

                return new CategoricalObservations(categoriesPyObj)
                {
                    Params = [ logitsPyObj ]
                };
            });
        }
    }
}