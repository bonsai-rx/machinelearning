using System;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// A lookup class for relating different <see cref="TransitionsType"/> to <see cref="TransitionsModel"/>
    /// and their corresponding Python string representations.
    /// </summary>
    public static class TransitionsLookup
    {
        private static readonly Dictionary<TransitionsType, (Type Type, string StringValue)> _lookup = new Dictionary<TransitionsType, (Type, string)>
        {
            { TransitionsType.Stationary, (typeof(StationaryTransitions), "stationary") },
            { TransitionsType.ConstrainedStationary, (typeof(ConstrainedStationaryTransitions), "constrained") },
            { TransitionsType.Sticky, (typeof(StickyTransitions), "sticky") },
            { TransitionsType.NeuralNetworkRecurrent, (typeof(NeuralNetworkRecurrentTransitions), "nn_recurrent") }
        };

        /// <summary>
        /// Gets the <see cref="Type"/> of the <see cref="TransitionsModel"/> corresponding to the given <see cref="TransitionsType"/>.
        /// </summary>
        public static Type GetTransitionsClassType(TransitionsType type) => _lookup[type].Type;

        /// <summary>
        /// Gets the Python string representation of the given <see cref="TransitionsType"/>.
        /// </summary>
        public static string GetString(TransitionsType type) => _lookup[type].StringValue;

        /// <summary>
        /// Gets the <see cref="TransitionsType"/> corresponding to the given Python string representation.
        /// </summary>
        public static TransitionsType GetFromString(string value) => _lookup.First(x => x.Value.StringValue == value).Key;

        /// <summary>
        /// Gets the <see cref="TransitionsType"/> corresponding to the given <see cref="Type"/> of <see cref="TransitionsModel"/> .
        /// </summary>
        public static TransitionsType GetFromType(Type type) => _lookup.First(x => x.Value.Type == type).Key;
    }
}
