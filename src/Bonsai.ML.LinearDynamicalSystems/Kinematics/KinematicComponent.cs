using System.ComponentModel;
using System.Xml.Serialization;
using Newtonsoft.Json;

namespace Bonsai.ML.LinearDynamicalSystems.Kinematics
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
        [JsonProperty("x_state_component")]
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
        [JsonProperty("y_state_component")]
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
        [JsonProperty("covariance")]
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
}
