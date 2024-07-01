using System;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    public static class ObservationsLookup
    {
        private static readonly Dictionary<ObservationType, (Type Type, string StringValue)> _lookup;

        static ObservationsLookup()
        {
            _lookup = new Dictionary<ObservationType, (Type, string)>
            {
                { ObservationType.Gaussian, (typeof(GaussianObservations), "gaussian") },
                { ObservationType.Exponential, (typeof(ExponentialObservations), "exponential") },
                { ObservationType.Autoregressive, (typeof(AutoRegressiveObservations), "ar") },
                { ObservationType.Bernoulli, (typeof(BernoulliObservations), "bernoulli") },
                { ObservationType.Poisson, (typeof(PoissonObservations), "poisson") },
            };
        }

        public static Type GetObservationsClassType(ObservationType type) => _lookup[type].Type;
        public static string GetString(ObservationType type) => _lookup[type].StringValue;
        public static ObservationType GetFromString(string value) => _lookup.First(x => x.Value.StringValue == value).Key;
        public static ObservationType GetFromType(Type type) => _lookup.First(x => x.Value.Type == type).Key;
    }
}
