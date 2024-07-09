using System;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    public static class ObservationsLookup
    {
        private static readonly Dictionary<ObservationsType, (Type Type, string StringValue)> _lookup;

        static ObservationsLookup()
        {
            _lookup = new Dictionary<ObservationsType, (Type, string)>
            {
                { ObservationsType.Gaussian, (typeof(GaussianObservations), "gaussian") },
                { ObservationsType.Exponential, (typeof(ExponentialObservations), "exponential") },
                { ObservationsType.Bernoulli, (typeof(BernoulliObservations), "bernoulli") },
                { ObservationsType.Poisson, (typeof(PoissonObservations), "poisson") },
                { ObservationsType.AutoRegressive, (typeof(AutoRegressiveObservations), "autoregressive") }
            };
        }

        public static Type GetObservationsClassType(ObservationsType type) => _lookup[type].Type;
        public static string GetString(ObservationsType type) => _lookup[type].StringValue;
        public static ObservationsType GetFromString(string value) => _lookup.First(x => x.Value.StringValue == value).Key;
        public static ObservationsType GetFromType(Type type) => _lookup.First(x => x.Value.Type == type).Key;
    }
}
