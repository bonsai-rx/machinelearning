using System;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// A lookup class for relating different <see cref="TransitionsModelType"/> to <see cref="TransitionsModel"/>
    /// and their corresponding Python string representations.
    /// </summary>
    public static class TransitionsModelLookup
    {
        private static readonly Dictionary<TransitionsModelType, (Type Type, string StringValue)> _lookup = new Dictionary<TransitionsModelType, (Type, string)>
        {
            { TransitionsModelType.Stationary, (typeof(StationaryTransitions), "stationary") },
            { TransitionsModelType.ConstrainedStationary, (typeof(ConstrainedStationaryTransitions), "constrained") },
            { TransitionsModelType.Sticky, (typeof(StickyTransitions), "sticky") },
            { TransitionsModelType.NeuralNetworkRecurrent, (typeof(NeuralNetworkRecurrentTransitions), "nn_recurrent") }
        };

        /// <summary>
        /// Gets the <see cref="Type"/> of the <see cref="TransitionsModel"/> corresponding to the given <see cref="TransitionsModelType"/>.
        /// </summary>
        public static Type GetTransitionsClassType(TransitionsModelType type) => _lookup[type].Type;

        /// <summary>
        /// Gets the Python string representation of the given <see cref="TransitionsModelType"/>.
        /// </summary>
        public static string GetString(TransitionsModelType type) => _lookup[type].StringValue;

        /// <summary>
        /// Gets the <see cref="TransitionsModelType"/> corresponding to the given Python string representation.
        /// </summary>
        public static TransitionsModelType GetFromString(string value) => _lookup.First(x => x.Value.StringValue == value).Key;

        /// <summary>
        /// Gets the <see cref="TransitionsModelType"/> corresponding to the given <see cref="Type"/> of <see cref="TransitionsModel"/> .
        /// </summary>
        public static TransitionsModelType GetFromType(Type type) => _lookup.First(x => x.Value.Type == type).Key;
    }
}
