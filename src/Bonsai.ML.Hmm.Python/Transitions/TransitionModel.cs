using System.ComponentModel;

namespace Bonsai.ML.Hmm.Python.Transitions
{
    /// <summary>
    /// An abstract class for creating a Transitions model.
    /// </summary>
    public abstract class TransitionModel : PythonModel
    {
        /// <summary>
        /// The type of Transitions model.
        /// </summary>
        public abstract TransitionModelType TransitionModelType { get; }

        /// <inheritdoc/>
        [Browsable(false)]
        protected override string ModelName => "transition";

        /// <inheritdoc/>
        [Browsable(false)]
        protected override string ModelType => TransitionModelLookup.GetString(TransitionModelType);

        /// <inheritdoc/>
        public TransitionModel() : base()
        {
        }

        /// <inheritdoc/>
        public TransitionModel(params object[] kwargs) : base(kwargs)
        {
        }
    }
}
