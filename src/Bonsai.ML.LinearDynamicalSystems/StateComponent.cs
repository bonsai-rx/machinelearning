using System.ComponentModel;
using YamlDotNet.Serialization;
using System;
using System.Xml.Serialization;
using System.Collections.Generic;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// State component of a Kalman Filter
    /// </summary>
    [Description("State component of a Kalman Filter")]  
    public class StateComponent
    {

        private double _mean;

        private double _variance;

        /// <summary>
        /// mean
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="mean")]
        [Description("mean")]
        public double Mean
        {
            get
            {
                return _mean;
            }
            set
            {
                _mean = value;
            }
        }

        /// <summary>
        /// variance
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="variance")]
        [Description("variance")]
        public double Variance
        {
            get
            {
                return _variance;
            }
            set
            {
                _variance = value;
            }
        }

        /// <summary>
        /// Extracts a single state compenent from the full state
        /// </summary>
        public StateComponent(List<List<double>> X, List<List<double>> P, int i) 
        {
            Mean = X[i][0];
            Variance = Sigma(P[i][i]);
        }

        private double Sigma(double variance)
        {
            return 2 * Math.Sqrt(variance);
        }
    }
}