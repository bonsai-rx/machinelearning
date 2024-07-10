using System.Text;

namespace Bonsai.ML.HiddenMarkovModels
{
    /// <summary>
    /// Provides a base class for building string representations of Python objects.
    /// </summary>
    public abstract class PythonStringBuilder
    {

        private string _cachedString;
        private bool _updateString;

        /// <summary>
        /// The internal string builder used to build the string representation.
        /// </summary>
        protected readonly StringBuilder StringBuilder = new StringBuilder();

        /// <summary>
        /// Sets a flag to update the string cache on the next call to the <see cref="ToString"/> method.
        /// </summary>
        protected void UpdateString()
        {
            _updateString = true;
        }

        /// <summary>
        /// Method used to build a string representation of the object.
        /// </summary>
        protected virtual string BuildString()
        {
            return StringBuilder.ToString();
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            if (_updateString)
            {
                _cachedString = BuildString();
                  _updateString = false;
            }
            return _cachedString;
        }
    }
}


