using System.ComponentModel;
using System;
using System.Reactive.Linq;
using System.Xml.Serialization;
using Newtonsoft.Json;

namespace Bonsai.ML.LinearDynamicalSystems.Kinematics
{
    /// <summary>
    /// State of a Kalman filter model representing kinematics of position, velocity, and acceleration
    /// </summary>
    [Description("State of a Kalman filter model representing kinematics of position, velocity, and acceleration")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class KinematicState
    {
        private KinematicComponent _position;
    
        private KinematicComponent _velocity;
    
        private KinematicComponent _acceleration;
    
        /// <summary>
        /// position kinematic component
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("position")]
        [Description("position kinematic component")]
        public KinematicComponent Position
        {
            get
            {
                return _position;
            }
            private set
            {
                _position = value;
            }
        }
    
        /// <summary>
        /// velocity kinematic components
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("velocity")]
        [Description("velocity kinematic components")]
        public KinematicComponent Velocity
        {
            get
            {
                return _velocity;
            }
            private set
            {
                _velocity = value;
            }
        }
    
        /// <summary>
        /// acceleration kinematic components
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("acceleration")]
        [Description("acceleration kinematic components")]
        public KinematicComponent Acceleration
        {
            get
            {
                return _acceleration;
            }
            private set
            {
                _acceleration = value;
            }
        }

                /// <summary>
        /// Converts the full state of a Kalman filter (mean vector and covariance matrix) into a KinematicState object representing position, velocity, and acceleration
        /// </summary>
        public IObservable<KinematicState> Process(IObservable<State> source)
        {
            return Observable.Select(source, state => 
            {
                KinematicComponent position = new KinematicComponent{
                    X = new StateComponent(state.X, state.P, 0),
                    Y = new StateComponent(state.X, state.P, 3),
                    Covariance = state.P[0,3]
                };

                KinematicComponent velocity = new KinematicComponent{
                    X = new StateComponent(state.X, state.P, 1),
                    Y = new StateComponent(state.X, state.P, 4),
                    Covariance = state.P[1,4]
                };

                KinematicComponent acceleration = new KinematicComponent{
                    X = new StateComponent(state.X, state.P, 2),
                    Y = new StateComponent(state.X, state.P, 5),
                    Covariance = state.P[2,5]
                };
                
                return new KinematicState {
                        Position = position,
                        Velocity = velocity,
                        Acceleration = acceleration
                    };
            });
        }
    }
}
