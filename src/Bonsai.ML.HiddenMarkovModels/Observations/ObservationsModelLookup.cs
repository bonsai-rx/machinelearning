using System;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// A lookup class for relating different <see cref="ObservationsModelType"/> to <see cref="ObservationsModel"/>
    /// and their corresponding Python string representations.
    /// </summary>
    public static class ObservationsModelLookup
    {
        private static readonly Dictionary<ObservationsModelType, (Type Type, string StringValue)> _lookup = new Dictionary<ObservationsModelType, (Type, string)>
        {
            { ObservationsModelType.Gaussian, (typeof(GaussianObservations), "gaussian") },
            { ObservationsModelType.Exponential, (typeof(ExponentialObservations), "exponential") },
            { ObservationsModelType.Bernoulli, (typeof(BernoulliObservations), "bernoulli") },
            { ObservationsModelType.Poisson, (typeof(PoissonObservations), "poisson") },
            { ObservationsModelType.AutoRegressive, (typeof(AutoRegressiveObservations), "autoregressive") },
            { ObservationsModelType.Categorical, (typeof(CategoricalObservations), "categorical") }
        };

        /// <summary>
        /// Gets the <see cref="Type"/> of the <see cref="ObservationsModel"/> corresponding to the given <see cref="ObservationsModelType"/>.
        /// </summary>
        public static Type GetObservationsClassType(ObservationsModelType type) => _lookup[type].Type;

        /// <summary>
        /// Gets the Python string representation of the given <see cref="ObservationsModelType"/>.
        /// </summary>
        public static string GetString(ObservationsModelType type) => _lookup[type].StringValue;

        /// <summary>
        /// Gets the <see cref="ObservationsModelType"/> corresponding to the given Python string representation.
        /// </summary>
        public static ObservationsModelType GetFromString(string value) => _lookup.First(x => x.Value.StringValue == value).Key;

        /// <summary>
        /// Gets the <see cref="ObservationsModelType"/> corresponding to the given <see cref="Type"/> of <see cref="ObservationsModel"/> .
        /// </summary>
        public static ObservationsModelType GetFromType(Type type) => _lookup.First(x => x.Value.Type == type).Key;
    }
}
