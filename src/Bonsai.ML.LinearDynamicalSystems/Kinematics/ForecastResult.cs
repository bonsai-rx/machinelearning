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
        /// <summary>
        /// kinematic state
        /// </summary>
        public KinematicState KinematicState { get; private set; }

        /// <summary>
        /// timestep
        /// </summary>
        public TimeSpan Timestep { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ForecastResult"/> class.
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
