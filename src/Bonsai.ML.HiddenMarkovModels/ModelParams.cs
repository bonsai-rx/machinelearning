using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;

namespace Bonsai.ML.HiddenMarkovModels
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class ModelParams
    {
        public int NumStates { get; set; }
        public int Dimensions { get; set; }
        public ObservationType ObservationType { get; set; }

        public IObservable<ModelParams> Process()
        {
            return Observable.Return(new ModelParams() { NumStates = NumStates, Dimensions = Dimensions, ObservationType = ObservationType });
        }

        public override string ToString()
        {
            observationTypeLookup.TryGetValue(ObservationType, out var observationType);
            return $"K={NumStates}, D={Dimensions}, observations={observationType}";
        }

        private static readonly Dictionary<ObservationType, string> observationTypeLookup = new Dictionary<ObservationType, string>
        {
            { ObservationType.Gaussian, "gaussian" },
            { ObservationType.Exponential, "exponential" },
            { ObservationType.Poisson, "poisson" },
            { ObservationType.Bernoulli, "bernoulli" },
            { ObservationType.Autoregressive, "ar" }
        };
    }
}

