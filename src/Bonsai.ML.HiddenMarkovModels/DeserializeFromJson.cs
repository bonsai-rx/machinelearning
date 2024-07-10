using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Bonsai.Expressions;
using System.Xml.Serialization;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Linq;
using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels
{
    /// <summary>
    /// Deserializes a sequence of JSON strings into data model objects.
    /// </summary>
    [DefaultProperty(nameof(Type))]
    [WorkflowElementCategory(ElementCategory.Transform)]
    [XmlInclude(typeof(TypeMapping<StateParameters>))]
    [XmlInclude(typeof(TypeMapping<ModelParameters>))]
    [Description("Deserializes a sequence of JSON strings into data model objects.")]
    public partial class DeserializeFromJson : SingleArgumentExpressionBuilder
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DeserializeFromJson"/> class.
        /// </summary>
        public DeserializeFromJson()
        {
            Type = new TypeMapping<StateParameters>();
        }

        /// <summary>
        /// Gets or sets the type of the object to deserialize.
        /// </summary>
        [Description("The type of the object to deserialize.")]
        public TypeMapping Type { get; set; }

        /// <inheritdoc/>
        public override Expression Build(IEnumerable<Expression> arguments)
        {
            TypeMapping typeMapping = Type;
            var returnType = typeMapping.GetType().GetGenericArguments()[0];
            return Expression.Call(
                typeof(DeserializeFromJson),
                nameof(Process),
                [ returnType ],
                Enumerable.Single(arguments));
        }

        private static IObservable<T> Process<T>(IObservable<string> source)
        {
            return source.Select(JsonConvert.DeserializeObject<T>);
        }
    }
}
