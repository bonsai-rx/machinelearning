using System;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// A lookup class for relating different <see cref="ObservationsType"/> to <see cref="ObservationsModel"/>
    /// and their corresponding Python string representations.
    /// </summary>
    public static class ObservationsLookup
    {
        private static readonly Dictionary<ObservationsType, (Type Type, string StringValue)> _lookup = new Dictionary<ObservationsType, (Type, string)>
        {
            { ObservationsType.Gaussian, (typeof(GaussianObservations), "gaussian") },
            { ObservationsType.Exponential, (typeof(ExponentialObservations), "exponential") },
            { ObservationsType.Bernoulli, (typeof(BernoulliObservations), "bernoulli") },
            { ObservationsType.Poisson, (typeof(PoissonObservations), "poisson") },
            { ObservationsType.AutoRegressive, (typeof(AutoRegressiveObservations), "autoregressive") }
        };

        /// <summary>
        /// Gets the <see cref="Type"/> of the <see cref="ObservationsModel"/> corresponding to the given <see cref="ObservationsType"/>.
        /// </summary>
        public static Type GetObservationsClassType(ObservationsType type) => _lookup[type].Type;

        /// <summary>
        /// Gets the Python string representation of the given <see cref="ObservationsType"/>.
        /// </summary>
        public static string GetString(ObservationsType type) => _lookup[type].StringValue;

        /// <summary>
        /// Gets the <see cref="ObservationsType"/> corresponding to the given Python string representation.
        /// </summary>
        public static ObservationsType GetFromString(string value) => _lookup.First(x => x.Value.StringValue == value).Key;

        /// <summary>
        /// Gets the <see cref="ObservationsType"/> corresponding to the given <see cref="Type"/> of <see cref="ObservationsModel"/> .
        /// </summary>
        public static ObservationsType GetFromType(Type type) => _lookup.First(x => x.Value.Type == type).Key;
    }
}
