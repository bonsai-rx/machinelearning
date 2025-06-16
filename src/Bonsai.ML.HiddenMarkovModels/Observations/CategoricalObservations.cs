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
    public class CategoricalObservations : ObservationModel
    {
        /// <summary>
        /// The number of categories in the observations.
        /// </summary>
        [Description("The number of categories in the observations.")]
        public int Categories { get; set; } = 2; 

        /// <summary>
        /// The logit of the observations for each state.
        /// </summary>
        [XmlIgnore]
        [Description("The logit of the observations for each state.")]
        public double[,,] Logits { get; set; } = null;

        /// <inheritdoc/>
        [JsonProperty]
        [JsonConverter(typeof(ObservationModelTypeJsonConverter))]
        [Browsable(false)]
        public override ObservationModelType ObservationModelType => ObservationModelType.Categorical;

        /// <inheritdoc/>
        [JsonProperty]
        public override object[] Params
        {
            get => [ Logits ];
        }

        /// <inheritdoc/>
        [JsonProperty]
        [XmlIgnore]
        [Browsable(false)]
        public override Dictionary<string, object> Kwargs => new Dictionary<string, object>
        {
            ["C"] = Categories,
        };

        /// <inheritdoc/>
        [XmlIgnore]
        [Browsable(false)]
        public static new string[] KwargsArray => [ "C" ];

        /// <inheritdoc/>
        public CategoricalObservations() : base()
        {
        }

        /// <inheritdoc/>
        public CategoricalObservations (params object[] kwargs) : base(kwargs)
        {
        }

        /// <inheritdoc/>
        protected override void CheckKwargs(params object[] kwargs)
        {
            if (kwargs is null || kwargs.Length != 1)
            {
                throw new ArgumentException($"The {nameof(CategoricalObservations)} operator requires exactly one keyword argument: {nameof(Categories)}.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateKwargs(params object[] kwargs)
        {
            Categories = kwargs[0] switch
            {
                int c => c,
                var c => Convert.ToInt32(c),
            };
        }

        /// <inheritdoc/>
        protected override void CheckParams(params object[] @params)
        {
            if (@params is not null && @params.Length != 1)
            {
                throw new ArgumentException($"The {nameof(CategoricalObservations)} operator requires exactly one parameter: {nameof(Logits)}.");
            }
        }

        /// <inheritdoc/>
        protected override void UpdateParams(params object[] @params)
        {
            Logits = (double[,,])@params[0];
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