using System.ComponentModel;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// An abstract class for creating an Observations model.
    /// </summary>
    public abstract class ObservationModel : PythonModel
    {
        /// <summary>
        /// The type of Observations model.
        /// </summary>
        public abstract ObservationModelType ObservationModelType { get; }

        /// <inheritdoc/>
        [Browsable(false)]
        protected override string ModelName => "observation";

        /// <inheritdoc/>
        [Browsable(false)]
        protected override string ModelType => ObservationModelLookup.GetString(ObservationModelType);

        /// <inheritdoc/>
        public ObservationModel() : base()
        {
        }

        /// <inheritdoc/>
        public ObservationModel(params object[] kwargs) : base(kwargs)
        {
        }
    }
}
