using System;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.Hmm.Python.Observations
{
    /// <summary>
    /// A lookup class for relating different <see cref="ObservationModelType"/> to <see cref="ObservationModel"/>
    /// and their corresponding Python string representations.
    /// </summary>
    public static class ObservationModelLookup
    {
        private static readonly Dictionary<ObservationModelType, (Type Type, string StringValue)> _lookup = new Dictionary<ObservationModelType, (Type, string)>
        {
            { ObservationModelType.Gaussian, (typeof(GaussianObservations), "gaussian") },
            { ObservationModelType.Exponential, (typeof(ExponentialObservations), "exponential") },
            { ObservationModelType.Bernoulli, (typeof(BernoulliObservations), "bernoulli") },
            { ObservationModelType.Poisson, (typeof(PoissonObservations), "poisson") },
            { ObservationModelType.AutoRegressive, (typeof(AutoRegressiveObservations), "autoregressive") },
            { ObservationModelType.Categorical, (typeof(CategoricalObservations), "categorical") }
        };

        /// <summary>
        /// Gets the <see cref="Type"/> of the <see cref="ObservationModel"/> corresponding to the given <see cref="ObservationModelType"/>.
        /// </summary>
        public static Type GetObservationsClassType(ObservationModelType type) => _lookup[type].Type;

        /// <summary>
        /// Gets the Python string representation of the given <see cref="ObservationModelType"/>.
        /// </summary>
        public static string GetString(ObservationModelType type) => _lookup[type].StringValue;

        /// <summary>
        /// Gets the <see cref="ObservationModelType"/> corresponding to the given Python string representation.
        /// </summary>
        public static ObservationModelType GetFromString(string value) => _lookup.First(x => x.Value.StringValue == value).Key;

        /// <summary>
        /// Gets the <see cref="ObservationModelType"/> corresponding to the given <see cref="Type"/> of <see cref="ObservationModel"/> .
        /// </summary>
        public static ObservationModelType GetFromType(Type type) => _lookup.First(x => x.Value.Type == type).Key;
    }
}
