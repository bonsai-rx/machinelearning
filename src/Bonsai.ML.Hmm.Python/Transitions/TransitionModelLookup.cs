using System;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.Hmm.Python.Transitions
{
    /// <summary>
    /// A lookup class for relating different <see cref="TransitionModelType"/> to <see cref="TransitionModel"/>
    /// and their corresponding Python string representations.
    /// </summary>
    public static class TransitionModelLookup
    {
        private static readonly Dictionary<TransitionModelType, (Type Type, string StringValue)> _lookup = new()
        {
            { TransitionModelType.Stationary, (typeof(StationaryTransitions), "stationary") },
            { TransitionModelType.ConstrainedStationary, (typeof(ConstrainedStationaryTransitions), "constrained") },
            { TransitionModelType.Sticky, (typeof(StickyTransitions), "sticky") },
            { TransitionModelType.NeuralNetworkRecurrent, (typeof(NeuralNetworkRecurrentTransitions), "nn_recurrent") }
        };

        /// <summary>
        /// Gets the <see cref="Type"/> of the <see cref="TransitionModel"/> corresponding to the given <see cref="TransitionModelType"/>.
        /// </summary>
        public static Type GetTransitionsClassType(TransitionModelType type) => _lookup[type].Type;

        /// <summary>
        /// Gets the Python string representation of the given <see cref="TransitionModelType"/>.
        /// </summary>
        public static string GetString(TransitionModelType type) => _lookup[type].StringValue;

        /// <summary>
        /// Gets the <see cref="TransitionModelType"/> corresponding to the given Python string representation.
        /// </summary>
        public static TransitionModelType GetFromString(string value) => _lookup.First(x => x.Value.StringValue == value).Key;

        /// <summary>
        /// Gets the <see cref="TransitionModelType"/> corresponding to the given <see cref="Type"/> of <see cref="TransitionModel"/> .
        /// </summary>
        public static TransitionModelType GetFromType(Type type) => _lookup.First(x => x.Value.Type == type).Key;
    }
}
