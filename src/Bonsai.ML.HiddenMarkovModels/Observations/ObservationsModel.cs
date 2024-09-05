using System.ComponentModel;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// An abstract class for creating an Observations model.
    /// </summary>
    public abstract class ObservationsModel : PythonModel
    {
        /// <summary>
        /// The type of Observations model.
        /// </summary>
        public abstract ObservationsModelType ObservationsModelType { get; }

        /// <inheritdoc/>
        [Browsable(false)]
        protected override string ModelName => "observations";

        /// <inheritdoc/>
        [Browsable(false)]
        protected override string ModelType => ObservationsModelLookup.GetString(ObservationsModelType);

        /// <inheritdoc/>
        public ObservationsModel() : base()
        {
        }

        /// <inheritdoc/>
        public ObservationsModel(params object[] kwargs) : base(kwargs)
        {
        }
    }
}
