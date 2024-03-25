using System;
using System.ComponentModel;
using System.Reactive.Linq;
using Bonsai.ML.LinearDynamicalSystems.Kinematics;
using Newtonsoft.Json;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// Serializes a sequence of data model objects into JSON strings.
    /// </summary>
    [Combinator]
    [WorkflowElementCategory(ElementCategory.Transform)]
    [Description("Serializes a sequence of data model objects into JSON strings.")]
    public class SerializeToJson
    {
        private IObservable<string> Process<T>(IObservable<T> source)
        {
            return source.Select(value => JsonConvert.SerializeObject(value));
        }

        /// <summary>
        /// Serializes each <see cref="KFModelParameters"/> object in the sequence to
        /// a JSON string.
        /// </summary>
        /// <param name="source">
        /// A sequence of <see cref="KFModelParameters"/> objects.
        /// </param>
        /// <returns>
        /// A sequence of JSON strings representing the corresponding
        /// <see cref="KFModelParameters"/> object.
        /// </returns>
        public IObservable<string> Process(IObservable<KFModelParameters> source)
        {
            return Process<KFModelParameters>(source);
        }

        /// <summary>
        /// Serializes each <see cref="Observation2D"/> object in the sequence to
        /// a JSON string.
        /// </summary>
        /// <param name="source">
        /// A sequence of <see cref="Observation2D"/> objects.
        /// </param>
        /// <returns>
        /// A sequence of JSON strings representing the corresponding
        /// <see cref="Observation2D"/> object.
        /// </returns>
        public IObservable<string> Process(IObservable<Observation2D> source)
        {
            return Process<Observation2D>(source);
        }

        /// <summary>
        /// Serializes each <see cref="State"/> object in the sequence to
        /// a JSON string.
        /// </summary>
        /// <param name="source">
        /// A sequence of <see cref="State"/> objects.
        /// </param>
        /// <returns>
        /// A sequence of JSON strings representing the corresponding
        /// <see cref="State"/> object.
        /// </returns>
        public IObservable<string> Process(IObservable<State> source)
        {
            return Process<State>(source);
        }

        /// <summary>
        /// Serializes each <see cref="StateComponent"/> object in the sequence to
        /// a JSON string.
        /// </summary>
        /// <param name="source">
        /// A sequence of <see cref="StateComponent"/> objects.
        /// </param>
        /// <returns>
        /// A sequence of JSON strings representing the corresponding
        /// <see cref="StateComponent"/> object.
        /// </returns>
        public IObservable<string> Process(IObservable<StateComponent> source)
        {
            return Process<StateComponent>(source);
        }

        /// <summary>
        /// Serializes each <see cref="KinematicState"/> object in the sequence to
        /// a JSON string.
        /// </summary>
        /// <param name="source">
        /// A sequence of <see cref="KinematicState"/> objects.
        /// </param>
        /// <returns>
        /// A sequence of JSON strings representing the corresponding
        /// <see cref="KinematicState"/> object.
        /// </returns>
        public IObservable<string> Process(IObservable<KinematicState> source)
        {
            return Process<KinematicState>(source);
        }

        /// <summary>
        /// Serializes each <see cref="KinematicComponent"/> object in the sequence to
        /// a JSON string.
        /// </summary>
        /// <param name="source">
        /// A sequence of <see cref="KinematicComponent"/> objects.
        /// </param>
        /// <returns>
        /// A sequence of JSON strings representing the corresponding
        /// <see cref="KinematicComponent"/> object.
        /// </returns>
        public IObservable<string> Process(IObservable<KinematicComponent> source)
        {
            return Process<KinematicComponent>(source);
        }

        public IObservable<string> Process(IObservable<LinearRegression.KFModelParameters> source)
        {
            return Process<LinearRegression.KFModelParameters>(source);
        }
    }
}
