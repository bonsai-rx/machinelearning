using System;
using System.Collections.Generic;
using System.Text;
using static Bonsai.ML.HiddenMarkovModels.Transitions.TransitionsLookup;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// An abstract class for creating an Transitions model.
    /// </summary>
    public abstract class TransitionsModel : PythonStringBuilder
    {
        /// <summary>
        /// The type of Transitions model.
        /// </summary>
        public abstract TransitionsType TransitionsType { get; }

        /// <summary>
        /// The parameters that are used to define the Transitions models.
        /// </summary>
        public abstract object[] Params { get; set; }

        /// <summary>
        /// The keyword arguments that are used to construct the Transitions models.
        /// </summary>
        public virtual Dictionary<string, object> Kwargs => new();

        /// <inheritdoc/>
        protected override string BuildString()
        {
            StringBuilder.Clear();
            StringBuilder.Append($"transition_type=\"{GetString(TransitionsType)}\"");

            if (Params != null && Params.Length > 0) 
            {
                var paramsStringBuilder = new StringBuilder();
                paramsStringBuilder.Append(",transition_params=(");

                foreach (var param in Params) {
                    if (param is null) {
                        paramsStringBuilder.Clear();
                        break;
                    }
                    var arrString = param is Array array ? NumpyHelper.NumpyParser.ParseArray(array) : param.ToString();
                    paramsStringBuilder.Append($"{arrString},");
                }

                if (paramsStringBuilder.Length > 0) {
                    paramsStringBuilder.Remove(paramsStringBuilder.Length - 1, 1);
                    paramsStringBuilder.Append(")");
                    StringBuilder.Append(paramsStringBuilder);
                }
                
            }

            if (Kwargs is not null && Kwargs.Count > 0)
            {
                StringBuilder.Append(",transition_kwargs={");
                foreach (var kp in Kwargs) {
                    StringBuilder.Append($"\"{kp.Key}\":{(kp.Value is null ? "None" 
                        : kp.Value is Array ? NumpyHelper.NumpyParser.ParseArray((Array)kp.Value) 
                        : kp.Value is string ? $"\"{kp.Value}\""
                        : kp.Value)},");
                }
                StringBuilder.Remove(StringBuilder.Length - 1, 1);
                StringBuilder.Append("}");   
            }

            return StringBuilder.ToString();
        }
    }
}
