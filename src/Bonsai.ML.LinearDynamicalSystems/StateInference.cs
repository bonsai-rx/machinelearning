namespace Bonsai.ML.LinearDynamicalSystems
{
    using System;
    using static PythonHelper;

    public class StateWithUncertainty
    {
        private double _x_state;

        private double _y_state;

        private double _x_uncertainty;

        private double _y_uncertainty;


        /// <summary>
        /// inferred x state
        /// </summary>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="x_state")]
        [System.ComponentModel.DescriptionAttribute("inferred X state")]
        public double X_state
        {
            get
            {
                return _x_state;
            }
            set
            {
                _x_state = value;
            }
        }

        /// <summary>
        /// inferred y state
        /// </summary>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="y_state")]
        [System.ComponentModel.DescriptionAttribute("inferred Y state")]
        public double Y_state
        {
            get
            {
                return _y_state;
            }
            set
            {
                _y_state = value;
            }
        }

        /// <summary>
        /// uncertainty in x state
        /// </summary>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="x_uncertainty")]
        [System.ComponentModel.DescriptionAttribute("uncertainty in X state")]
        public double X_uncertainty
        {
            get
            {
                return _x_uncertainty;
            }
            set
            {
                _x_uncertainty = value;
            }
        }

        /// <summary>
        /// uncertainty in y state
        /// </summary>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="y_uncertainty")]
        [System.ComponentModel.DescriptionAttribute("uncertainty in Y state")]
        public double Y_uncertainty
        {
            get
            {
                return _y_uncertainty;
            }
            set
            {
                _y_uncertainty = value;
            }
        }

        public StateWithUncertainty(double x_state, double y_state, double x_uncertainty, double y_uncertainty)
        {
            X_state = x_state;
            Y_state = y_state;
            X_uncertainty = x_uncertainty;
            Y_uncertainty = y_uncertainty;
        }
    }

    /// <summary>
    /// Infered state variables of a Kalman Filter python class (position, velocity, and acceleration)
    /// </summary>
    [System.ComponentModel.DescriptionAttribute("Infered state variables of a Kalman Filter python class (position, velocity, and acceleration)")]    
    public partial class StateEstimate
    {
    
        private StateWithUncertainty _position;
    
        private StateWithUncertainty _velocity;
    
        private StateWithUncertainty _acceleration;
    
        /// <summary>
        /// inferred position
        /// </summary>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="position")]
        [System.ComponentModel.DescriptionAttribute("inferred position")]
        public StateWithUncertainty Position
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
        /// inferred velocity
        /// </summary>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="velocity")]
        [System.ComponentModel.DescriptionAttribute("inferred velocity")]
        public StateWithUncertainty Velocity
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
        /// inferred acceleration
        /// </summary>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="acceleration")]
        [System.ComponentModel.DescriptionAttribute("inferred acceleration")]
        public StateWithUncertainty Acceleration
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
    /// StateInferenceed state variables of a Kalman Filter python class (position, velocity, and acceleration)
    /// </summary>
    [System.ComponentModel.DescriptionAttribute("Infered state variables from state")]    
    [Bonsai.CombinatorAttribute()]
    [Bonsai.WorkflowElementCategoryAttribute(Bonsai.ElementCategory.Source)]
    public class StateInference
    {
        private double SigmaFromVariance(double variance)
        {
            return 2 * Math.Sqrt(variance);
        }

        public System.IObservable<StateEstimate> Process(System.IObservable<State> source)
        {
            return System.Reactive.Linq.Observable.Select(source, state => 
            {
                StateWithUncertainty position = new StateWithUncertainty(
                    state.X[0][0],
                    state.X[3][0],
                    SigmaFromVariance(state.P[0][0]),
                    SigmaFromVariance(state.P[3][3])
                );

                StateWithUncertainty velocity = new StateWithUncertainty(
                    state.X[1][0],
                    state.X[4][0],
                    SigmaFromVariance(state.P[1][1]),
                    SigmaFromVariance(state.P[4][4])
                );

                StateWithUncertainty acceleration = new StateWithUncertainty(
                    state.X[2][0],
                    state.X[5][0],
                    SigmaFromVariance(state.P[2][2]),
                    SigmaFromVariance(state.P[5][5])
                );
                
                return new StateEstimate {
                        Position = position,
                        Velocity = velocity,
                        Acceleration = acceleration
                    };
            });
        }
    }
}