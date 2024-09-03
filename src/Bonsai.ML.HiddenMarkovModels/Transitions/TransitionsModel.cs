using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Bonsai.ML.Python;
using System.Xml.Serialization;
using System.Linq;
using System.ComponentModel;

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
        [Browsable(false)]
        public abstract TransitionsModelType TransitionsModelType { get; }

        /// <summary>
        /// The parameters that are used to define the Transitions models.
        /// </summary>
        [Browsable(false)]
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
        /// The array of keywords used to construct the Transitions models.
        /// </summary>
        [Browsable(false)]
        public static string[] KwargsArray => null;

        /// <summary>
        /// The dictionary of keyword arguments that are used to construct the Transitions models.
        /// </summary>
        [Browsable(false)]
        [XmlIgnore]
        public virtual Dictionary<string, object> Kwargs => new();

        /// <summary>
        /// Checks the keyword arguments.
        /// </summary>
        /// <param name="kwargs">The keyword arguments.</param>
        protected virtual void CheckKwargs(params object[] kwargs)
        {
        }

        /// <summary>
        /// Updates the keyword arguments of the Transitions model.
        /// </summary>
        /// <param name="kwargs">The keyword arguments.</param>
        protected virtual void UpdateKwargs(params object[] kwargs)
        {
        }

        /// <summary>
        /// Checks the parameters.
        /// </summary>
        /// <param name="params">The parameters.</param>
        protected virtual void CheckParams(params object[] @params)
        {
        }

        /// <summary>
        /// Updates the parameters of the Transitions model.
        /// </summary>
        /// <param name="params">The parameters.</param>
        protected virtual void UpdateParams(params object[] @params)
        {
        }

        /// <summary>
        /// Constructs a new instance of the <see cref="TransitionsModel"/> class.
        /// </summary>
        public TransitionsModel()
        {
            BuildString();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TransitionsModel"/> class using keyword arguments.
        /// </summary>
        /// <param name="kwargs">The keyword arguments.</param>
        public TransitionsModel(params object[] kwargs)
        {
            CheckKwargs(kwargs);
            UpdateKwargs(kwargs);
            UpdateString();
        }

        /// <inheritdoc/>
        protected override string BuildString()
        {
            StringBuilder.Clear();
            StringBuilder.Append($"transitions_model_type=\"{TransitionsModelLookup.GetString(TransitionsModelType)}\"");

            if (Params != null && Params.Length > 0 && Params.All(p => p != null)) 
            {
                var paramsStringBuilder = new StringBuilder();

                foreach (var param in Params) 
                {
                    if (param is null) 
                    {
                        paramsStringBuilder.Append("None");
                    }
                    else if (param is Array array) 
                    {
                        paramsStringBuilder.Append(NumpyHelper.NumpyParser.ParseArray(array));
                    } 
                    else if (param is IList list) 
                    {
                        if (list.Count == 0) 
                        {
                            paramsStringBuilder.Append("[]");
                        }
                        else
                        {
                            paramsStringBuilder.Append("[");
                            foreach (var item in list) 
                            {
                                paramsStringBuilder.Append(item is Array itemArray ? NumpyHelper.NumpyParser.ParseArray(itemArray) : item.ToString());
                                paramsStringBuilder.Append(",");
                            }
                            paramsStringBuilder.Remove(paramsStringBuilder.Length - 1, 1);
                            paramsStringBuilder.Append("]");
                        }
                    } 
                    else 
                    {
                        paramsStringBuilder.Append(param.ToString());
                    }

                    paramsStringBuilder.Append(",");
                }

                paramsStringBuilder.Remove(paramsStringBuilder.Length - 1, 1);
                StringBuilder.Append($",transition_params=({paramsStringBuilder})");
                
            }

            if (Kwargs is not null && Kwargs.Count > 0)
            {
                StringBuilder.Append(",transition_kwargs={");
                foreach (var kp in Kwargs) {
                    StringBuilder.Append($"\"{kp.Key}\":{(kp.Value is null ? "None" 
                        : kp.Value is Array array ? NumpyHelper.NumpyParser.ParseArray(array)
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
