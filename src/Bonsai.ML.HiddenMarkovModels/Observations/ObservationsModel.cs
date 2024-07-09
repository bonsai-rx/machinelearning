using Newtonsoft.Json;
using System;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// An abstract class for creating an observations model.
    /// </summary>
    public abstract class ObservationsModel
    {
        /// <summary>
        /// The type of observations model.
        /// </summary>
        public abstract ObservationsType ObservationsType { get; }

        /// <summary>
        /// The parameters that are used to define the observations models.
        /// </summary>
        public abstract object[] Params { get; set; }

        /// <summary>
        /// The keyword arguments that are used to construct the observations models.
        /// </summary>
        public virtual object[] Kwargs => Array.Empty<object>();
    }
}
