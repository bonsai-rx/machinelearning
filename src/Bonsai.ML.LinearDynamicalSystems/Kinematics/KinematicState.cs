using System.ComponentModel;
using System;
using System.Reactive.Linq;
using System.Xml.Serialization;
using Newtonsoft.Json;

namespace Bonsai.ML.LinearDynamicalSystems.Kinematics
{
    /// <summary>
    /// Represents an operator that converts the full state of a Kalman filter model into a KinematicState class representing position, velocity, and acceleration.
    /// </summary>
    [Combinator]
    [Description("Converts the full state of a Kalman filter model into a KinematicState representing position, velocity, and acceleration.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class KinematicState
    {    
        /// <summary>
        /// Gets or sets the position kinematic component.
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("position")]
        [Description("The position kinematic component")]
        public KinematicComponent Position { get; set; }
    
        /// <summary>
        /// Gets or sets the velocity kinematic component.
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("velocity")]
        [Description("THe velocity kinematic component")]
        public KinematicComponent Velocity { get; set; }
    
        /// <summary>
        /// Gets or sets the acceleration kinematic component.
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("acceleration")]
        [Description("The acceleration kinematic component")]
        public KinematicComponent Acceleration { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="KinematicState"/> class
        /// </summary>
        public KinematicState ()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="KinematicState"/> class
        /// from the full state of a Kalman filter model.
        /// </summary>
        public KinematicState (State state)
        {
            Position = new KinematicComponent{
                X = new StateComponent(state.X, state.P, 0),
                Y = new StateComponent(state.X, state.P, 3),
                Covariance = state.P[0,3]
            };

            Velocity = new KinematicComponent{
                X = new StateComponent(state.X, state.P, 1),
                Y = new StateComponent(state.X, state.P, 4),
                Covariance = state.P[1,4]
            };

            Acceleration = new KinematicComponent{
                X = new StateComponent(state.X, state.P, 2),
                Y = new StateComponent(state.X, state.P, 5),
                Covariance = state.P[2,5]
            };
        }

        /// <summary>
        /// Converts the full state of a Kalman filter (mean vector and covariance matrix) into a KinematicState object representing position, velocity, and acceleration
        /// </summary>
        public IObservable<KinematicState> Process(IObservable<State> source)
        {
            return Observable.Select(source, state => 
            {
                return new KinematicState(state);
            });
        }
    }
}
