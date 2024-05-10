using System;
using Newtonsoft.Json;
using Python.Runtime;
using System.Collections.Generic;
using System.Xml.Serialization;
using System.ComponentModel;

namespace Bonsai.ML.LinearDynamicalSystems.Kinematics
{
    /// <summary>
    /// Forecast result representing a collection of forecasted states at future timesteps
    /// </summary>
    public class ForecastResult
    {
        private KinematicState kinematicState;

        private TimeSpan timestep;

        /// <summary>
        /// kinematic state
        /// </summary>
        public KinematicState KinematicState
        {
            get
            {
                return kinematicState;
            }
            private set
            {
                kinematicState = value;
            }
        }

        /// <summary>
        /// timestep
        /// </summary>
        public TimeSpan Timestep
        {
            get
            {
                return timestep;
            }
            private set
            {
                timestep = value;
            }
        }

        /// <summary>
        /// Constructor of a forecast result class
        /// </summary>
        /// <param name="kinematicState"></param>
        /// <param name="timestep"></param>
        public ForecastResult(KinematicState kinematicState, TimeSpan timestep)
        {
            KinematicState = kinematicState;
            Timestep = timestep;
        }
    }
}
