using System;
using System.Collections.Generic;
using System.Text;
using Bonsai.ML.Data;
using System.Xml.Serialization;
using System.Linq;
using System.Reactive.Linq;
using System.ComponentModel;

namespace Bonsai.ML.HiddenMarkovModels
{
    /// <summary>
    /// An abstract class for creating a Python model.
    /// </summary>
    public abstract class PythonModel : PythonStringBuilder
    {
        /// <summary>
        /// The parameters that are used to define the model.
        /// </summary>
        [Browsable(false)]
        [XmlIgnore]
        public virtual object[] Params 
        { 
            get => null; 
            set
            {
                CheckParams(value);
                UpdateParams(value);
                UpdateString();
            }
        }

        /// <summary>
        /// The array of keyword arguments used to construct the model.
        /// </summary>
        public static string[] KwargsArray => null;

        /// <summary>
        /// The dictionary of keyword arguments that are used to construct the model.
        /// </summary>
        [Browsable(false)]
        [XmlIgnore]
        public virtual Dictionary<string, object> Kwargs => new();

        /// <summary>
        /// Checks if the keyword arguments are valid. 
        /// </summary>
        /// <param name="kwargs">The keyword arguments.</param>
        protected virtual void CheckKwargs(params object[] kwargs)
        {
        }

        /// <summary>
        /// Updates the kwargs dictionary.
        /// </summary>
        /// <param name="kwargs">The keyword arguments.</param>
        protected virtual void UpdateKwargs(params object[] kwargs)
        {
        }

        /// <summary>
        /// Checks if the parameters are valid.
        /// </summary>
        /// <param name="params">The parameters.</param>
        protected virtual void CheckParams(params object[] @params)
        {
        }

        /// <summary>
        /// Updates the parameters.
        /// </summary>
        /// <param name="params">The parameters.</param>
        protected virtual void UpdateParams(params object[] @params)
        {
        }

        /// <summary>
        /// Constructs a new instance of the <see cref="PythonModel"/> class.
        /// </summary>
        public PythonModel()
        {
            BuildString();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PythonModel"/> class using keyword arguments.
        /// </summary>
        /// <param name="kwargs">The keyword arguments.</param>
        public PythonModel(params object[] kwargs)
        {
            CheckKwargs(kwargs);
            UpdateKwargs(kwargs);
            UpdateString();
        }

        /// <summary>
        /// The name of the base python model class.
        /// </summary>
        protected abstract string ModelName { get; }

        /// <summary>
        /// The specific type of the model.
        /// </summary>
        protected abstract string ModelType { get; }

        /// <inheritdoc/>
        protected override string BuildString()
        {
            // StringBuilder.Clear();
            StringBuilder.Append($"{ModelName}_model_type=\"{ModelType}\"");

            if (Params != null && Params.Length > 0 && Params.All(p => p != null)) 
            {
                var paramsStringBuilder = new StringBuilder();
                paramsStringBuilder.Append($",{ModelName}_params=(");

                foreach (var param in Params) {
                    if (param is null) {
                        paramsStringBuilder.Clear();
                        break;
                    }
                    var arrString = param is Array array ? ArrayHelper.SerializeArrayToJson(array) : param.ToString();
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
                StringBuilder.Append($",{ModelName}_kwargs={{");
                foreach (var kp in Kwargs) {
                    StringBuilder.Append($"\"{kp.Key}\":{(kp.Value is null ? "None" 
                        : kp.Value is Array array ? ArrayHelper.SerializeArrayToJson(array)
                        : kp.Value is string ? $"\"{kp.Value}\""
                        : kp.Value)},");
                }
                StringBuilder.Remove(StringBuilder.Length - 1, 1);
                StringBuilder.Append("}");   
            }

            var result = StringBuilder.ToString();
            StringBuilder.Clear();
            return result;
        }
    }
}
