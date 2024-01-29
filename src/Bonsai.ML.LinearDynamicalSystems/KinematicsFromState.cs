using System.ComponentModel;
using YamlDotNet.Serialization;
using System;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// State components of a Kalman Filter Kinematics model
    /// </summary>
    [Description("State component of a Kalman Filter Kinematics model")]  
    public class KinematicComponent
    {
        private StateComponent _x;

        private StateComponent _y;

        private double _covariance;

        /// <summary>
        /// x state component
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="x_state_component")]
        [Description("X state component")]
        public StateComponent X
        {
            get
            {
                return _x;
            }
            set
            {
                _x = value;
            }
        }

        /// <summary>
        /// y state component
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="y_state_component")]
        [Description("Y state component")]
        public StateComponent Y
        {
            get
            {
                return _y;
            }
            set
            {
                _y = value;
            }
        }

        /// <summary>
        /// covariance between state components
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="covariance")]
        [Description("covariance between state components")]
        public double Covariance
        {
            get
            {
                return _covariance;
            }
            set
            {
                _covariance = value;
            }
        }
    }

    /// <summary>
    /// Kinematics of position, velocity, and acceleration
    /// </summary>
    [Description("Kinematics of position, velocity, and acceleration")]    
    public class Kinematics
    {
        private KinematicComponent _position;
    
        private KinematicComponent _velocity;
    
        private KinematicComponent _acceleration;
    
        /// <summary>
        /// position kinematic component
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="position")]
        [Description("position kinematic component")]
        public KinematicComponent Position
        {
            get
            {
                return _position;
            }
            set
            {
                _position = value;
            }
        }
    
        /// <summary>
        /// velocity kinematic components
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="velocity")]
        [Description("velocity kinematic components")]
        public KinematicComponent Velocity
        {
            get
            {
                return _velocity;
            }
            set
            {
                _velocity = value;
            }
        }
    
        /// <summary>
        /// acceleration kinematic components
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="acceleration")]
        [Description("acceleration kinematic components")]
        public KinematicComponent Acceleration
        {
            get
            {
                return _acceleration;
            }
            set
            {
                _acceleration = value;
            }
        }
    }

    /// <summary>
    /// Kinematic components grabbed from the state of a Kalman Filter model
    /// </summary>
    [Description("Kinematic components grabbed from the state of a Kalman Filter model")]    
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class KinematicsFromState
    {

        /// <summary>
        /// Converts the full state of a Kalman filter (mean vector and covariance matrix) into a Kinematics object representing position, velocity, and acceleration
        /// </summary>
        public IObservable<Kinematics> Process(IObservable<State> source)
        {
            return Observable.Select(source, state => 
            {
                KinematicComponent position = new KinematicComponent{
                    X = new StateComponent(state.X, state.P, 0),
                    Y = new StateComponent(state.X, state.P, 3),
                    Covariance = state.P[0][3]
                };

                KinematicComponent velocity = new KinematicComponent{
                    X = new StateComponent(state.X, state.P, 1),
                    Y = new StateComponent(state.X, state.P, 4),
                    Covariance = state.P[1][4]
                };

                KinematicComponent acceleration = new KinematicComponent{
                    X = new StateComponent(state.X, state.P, 2),
                    Y = new StateComponent(state.X, state.P, 5),
                    Covariance = state.P[2][5]
                };
                
                return new Kinematics {
                        Position = position,
                        Velocity = velocity,
                        Acceleration = acceleration
                    };
            });
        }
    }
}