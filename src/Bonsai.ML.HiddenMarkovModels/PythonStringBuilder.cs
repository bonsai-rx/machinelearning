using System;
using System.Text;

namespace Bonsai.ML.HiddenMarkovModels
{
    public abstract class PythonStringBuilder
    {

        private string _cachedString;
        private bool _updateString;
        protected readonly StringBuilder StringBuilder = new StringBuilder();

        protected void UpdateString()
        {
            _updateString = true;
        }

        protected virtual string BuildString()
        {
            return StringBuilder.ToString();
        }

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


