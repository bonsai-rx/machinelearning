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
        /// Gets or privately sets the kinematic state of the forecasted result.
        /// </summary>
        public KinematicState KinematicState { get; private set; }

        /// <summary>
        /// Gets or privately sets the future time step of the forecasted result.
        /// </summary>
        public TimeSpan Timestep { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ForecastResult"/> class.
        /// </summary>
        /// <param name="kinematicState">The kinematic state of the forecasted result.</param>
        /// <param name="timestep">The future timestep of the forecasted result.</param>
        public ForecastResult(KinematicState kinematicState, TimeSpan timestep)
        {
            KinematicState = kinematicState;
            Timestep = timestep;
        }
    }
}
