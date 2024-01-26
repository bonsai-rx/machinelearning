using System.ComponentModel;
using YamlDotNet.Serialization;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Reflection;
using System.Xml.Serialization;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.LinearDynamicalSystems
{

    /// <summary>
    /// State components of a Kalman Filter Kinematics model
    /// </summary>
    [Description("State component of a Kalman Filter Kinematics model")]  
    public class KinematicComponent
    {
        private double _x_mean;

        private double _y_mean;

        private double _x_variance;

        private double _y_variance;


        /// <summary>
        /// x mean
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="x_mean")]
        [Description("X mean")]
        public double X_mean
        {
            get
            {
                return _x_mean;
            }
            set
            {
                _x_mean = value;
            }
        }

        /// <summary>
        /// y mean
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="y_mean")]
        [Description("Y mean")]
        public double Y_mean
        {
            get
            {
                return _y_mean;
            }
            set
            {
                _y_mean = value;
            }
        }

        /// <summary>
        /// x variance
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="x_variance")]
        [Description("x variance")]
        public double X_variance
        {
            get
            {
                return _x_variance;
            }
            set
            {
                _x_variance = value;
            }
        }

        /// <summary>
        /// y variance
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="y_variance")]
        [Description("y variance")]
        public double Y_variance
        {
            get
            {
                return _y_variance;
            }
            set
            {
                _y_variance = value;
            }
        }
    }

    /// <summary>
    /// Kinematic variables of position, velocity, and acceleration
    /// </summary>
    [Description("Kinematic variables of position, velocity, and acceleration")]    
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
        private double Sigma(double variance)
        {
            return 2 * Math.Sqrt(variance);
        }

        public IObservable<Kinematics> Process(IObservable<State> source)
        {
            return Observable.Select(source, state => 
            {
                KinematicComponent position = new KinematicComponent{
                    X_mean = state.X[0][0],
                    Y_mean = state.X[3][0],
                    X_variance = Sigma(state.P[0][0]),
                    Y_variance = Sigma(state.P[3][3])
                };

                KinematicComponent velocity = new KinematicComponent{
                    X_mean = state.X[1][0],
                    Y_mean = state.X[4][0],
                    X_variance = Sigma(state.P[1][1]),
                    Y_variance = Sigma(state.P[4][4])
                };

                KinematicComponent acceleration = new KinematicComponent{
                    X_mean = state.X[2][0],
                    Y_mean = state.X[5][0],
                    X_variance = Sigma(state.P[2][2]),
                    Y_variance = Sigma(state.P[5][5])
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