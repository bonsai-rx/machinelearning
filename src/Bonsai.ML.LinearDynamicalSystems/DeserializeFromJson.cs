using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Bonsai.Expressions;
using System.Xml.Serialization;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Linq;
using Newtonsoft.Json;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// Deserializes a sequence of JSON strings into data model objects.
    /// </summary>
    [DefaultProperty("Type")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    [XmlInclude(typeof(TypeMapping<Kinematics.KFModelParameters>))]
    [XmlInclude(typeof(TypeMapping<Kinematics.Observation2D>))]
    [XmlInclude(typeof(TypeMapping<State>))]
    [XmlInclude(typeof(TypeMapping<StateComponent>))]
    [XmlInclude(typeof(TypeMapping<Kinematics.KinematicState>))]
    [XmlInclude(typeof(TypeMapping<Kinematics.KinematicComponent>))]
    [Description("Deserializes a sequence of JSON strings into data model objects.")]
    public partial class DeserializeFromJson : SingleArgumentExpressionBuilder
    {
        public DeserializeFromJson()
        {
            Type = new TypeMapping<Kinematics.KFModelParameters>();
        }

        public TypeMapping Type { get; set; }

        public override Expression Build(IEnumerable<Expression> arguments)
        {
            TypeMapping typeMapping = Type;
            var returnType = typeMapping.GetType().GetGenericArguments()[0];
            return Expression.Call(
                typeof(DeserializeFromJson),
                nameof(Process),
                new Type[] { returnType },
                Enumerable.Single(arguments));
        }

        private static IObservable<T> Process<T>(IObservable<string> source)
        {
            return source.Select(value => JsonConvert.DeserializeObject<T>(value));
        }
    }
}
