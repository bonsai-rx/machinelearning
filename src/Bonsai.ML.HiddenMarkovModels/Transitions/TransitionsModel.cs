using System;
using System.Collections.Generic;
using System.Text;
using Bonsai.ML.Python;

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
        public abstract TransitionsModelType TransitionsModelType { get; }

        /// <summary>
        /// The parameters that are used to define the Transitions models.
        /// </summary>
        public virtual object[] Params { get; set; }

        /// <summary>
        /// The keyword arguments that are used to construct the Transitions models.
        /// </summary>
        public virtual Dictionary<string, object> Kwargs => new();

        /// <summary>
        /// Checks the constructor parameters.
        /// </summary>
        /// <param name="args">The constructor parameters.</param>
        /// <returns>True if the constructor parameters are valid; otherwise, false.</returns>
        protected virtual bool CheckConstructorArgs(params object[] args)
        {
            return true;
        }

        /// <summary>
        /// Updates the keyword arguments of the observations model.
        /// </summary>
        /// <param name="kwargs">The keyword arguments.</param>
        protected virtual void UpdateKwargs(params object[] kwargs)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TransitionsModel"/> class.
        /// </summary>
        /// <param name="args">The constructor parameters.</param>
        protected TransitionsModel(params object[] args)
        {
            CheckConstructorArgs(args);
            UpdateKwargs(args);
            UpdateString();
        }

        /// <inheritdoc/>
        protected override string BuildString()
        {
            StringBuilder.Clear();
            StringBuilder.Append($"transitions_model_type=\"{TransitionsModelLookup.GetString(TransitionsModelType)}\"");

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
