using System.ComponentModel;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// An abstract class for creating a Transitions model.
    /// </summary>
    public abstract class TransitionsModel : PythonModel
    {
        /// <summary>
        /// The type of Transitions model.
        /// </summary>
        public abstract TransitionsModelType TransitionsModelType { get; }

        /// <inheritdoc/>
        [Browsable(false)]
        protected override string ModelName => "transitions";

        /// <inheritdoc/>
        [Browsable(false)]
        protected override string ModelType => TransitionsModelLookup.GetString(TransitionsModelType);

        /// <inheritdoc/>
        public TransitionsModel() : base()
        {
        }

        /// <inheritdoc/>
        public TransitionsModel(params object[] kwargs) : base(kwargs)
        {
        }
    }
}
