using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using System.Linq.Expressions;
using System.Reflection;
using Bonsai.Expressions;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Converts the input tensor into a flattened array of the specified element type.
    /// </summary>
    [Combinator]
    [Description("Converts the input tensor into a flattened array of the specified element type.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ToArray : SingleArgumentExpressionBuilder
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ToArray"/> class.
        /// </summary>
        public ToArray()
        {
            Type = typeof(double);
        }

        /// <summary>
        /// Gets or sets the type mapping used to convert the input tensor into a flattened array.
        /// </summary>
        [Description("Gets or sets the type mapping used to convert the input tensor into a flattened array.")]
        [TypeConverter(typeof(ScalarTypeConverter))]
        [XmlIgnore]
        public Type Type { get; set; }

        /// <summary>
        /// Gets or sets an XML serializable representation of the type.
        /// </summary>
        [Browsable(false)]
        [XmlElement("Type")]
        [EditorBrowsable(EditorBrowsableState.Never)]
        public string XmlType 
        {
            get { return Type.AssemblyQualifiedName; }
            set { Type = Type.GetType(value); }
        }

        /// <inheritdoc/>
        public override Expression Build(IEnumerable<Expression> arguments)
        {
            MethodInfo methodInfo = GetType().GetMethod("Process", BindingFlags.Public | BindingFlags.Instance);
            methodInfo = methodInfo.MakeGenericMethod(Type);
            Expression sourceExpression = arguments.First();
            
            return Expression.Call(
                Expression.Constant(this),
                methodInfo,
                sourceExpression
            );
        }

        /// <summary>
        /// Converts the input tensor into a flattened array of the specified element type.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<T[]> Process<T>(IObservable<Tensor> source) where T : unmanaged
        {
            return source.Select(tensor => tensor.data<T>().ToArray());
        }
    }
}