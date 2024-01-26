using System.ComponentModel;
using YamlDotNet.Serialization;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Reflection;
using Bonsai.Expressions;
using System.Xml.Serialization;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Linq;
using System.IO;
using YamlDotNet.Core;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// Deserializes a sequence of YAML strings into data model objects.
    /// </summary>
    [DefaultProperty("Type")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    [XmlInclude(typeof(TypeMapping<KalmanFilterKinematicsModel>))]
    [XmlInclude(typeof(TypeMapping<Observation2D>))]
    [XmlInclude(typeof(TypeMapping<State>))]
    [XmlInclude(typeof(TypeMapping<Kinematics>))]
    [XmlInclude(typeof(TypeMapping<KinematicComponent>))]
    [Description("Deserializes a sequence of YAML strings into data model objects.")]
    public partial class DeserializeFromYaml : SingleArgumentExpressionBuilder
    {
    
        public DeserializeFromYaml()
        {
            Type = new TypeMapping<KalmanFilterKinematicsModel>();
        }

        public TypeMapping Type { get; set; }

        public override Expression Build(IEnumerable<Expression> arguments)
        {
            TypeMapping typeMapping = Type;
            var returnType = typeMapping.GetType().GetGenericArguments()[0];
            return Expression.Call(
                typeof(DeserializeFromYaml),
                "Process",
                new Type[] { returnType },
                Enumerable.Single(arguments));
        }

        private static IObservable<T> Process<T>(IObservable<string> source)
        {
            return Observable.Defer(() =>
            {
                var serializer = new DeserializerBuilder().Build();
                return Observable.Select(source, value =>
                {
                    var reader = new StringReader(value);
                    var parser = new MergingParser(new Parser(reader));
                    return serializer.Deserialize<T>(parser);
                });
            });
        }
    }
}