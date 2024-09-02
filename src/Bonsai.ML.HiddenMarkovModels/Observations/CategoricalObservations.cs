using System;
using Python.Runtime;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Collections.Generic;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.Python;

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
    public class CategoricalObservations : ObservationsModel
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

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(ObservationsModelTypeJsonConverter))]
        public override ObservationsModelType ObservationsModelType => ObservationsModelType.Categorical;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get =>[ Logits ];
            set
            {
                Logits = (double[,,])value[0];
                UpdateString();
            }
        }

        /// <inheritdoc/>
        [JsonProperty]
        public override Dictionary<string, object> Kwargs => new Dictionary<string, object>
        {
            ["C"] = Categories,
        };

        /// <summary>
        /// Initializes a new instance of the <see cref="CategoricalObservations"/> class.
        /// </summary>
        public CategoricalObservations (params object[] args) : base(args)
        {
        }

        /// <inheritdoc/>
        protected override bool CheckConstructorArgs(params object[] args)
        {
            if (args is null || args.Length != 1)
            {
                throw new ArgumentException("The CategoricalObservations operator requires a single argument specifying the number of categories.");
            }
            return true;
        }

        /// <inheritdoc/>
        protected override void UpdateKwargs(params object[] args)
        {
            Categories = args[0] switch
            {
                int c => c,
                var c => Convert.ToInt32(c),
            };
        }

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