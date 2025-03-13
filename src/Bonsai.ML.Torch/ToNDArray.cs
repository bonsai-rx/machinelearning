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
    /// Converts the input tensor into an array of the specified element type and rank.
    /// </summary>
    [Combinator]
    [Description("Converts the input tensor into an array of the specified element type.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ToNDArray : SingleArgumentExpressionBuilder
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ToNDArray"/> class.
        /// </summary>
        public ToNDArray()
        {
            Type = typeof(double);
        }

        /// <summary>
        /// Gets or sets the type mapping used to convert the input tensor into an array.
        /// </summary>
        [Description("Gets or sets the type mapping used to convert the input tensor into an array.")]
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

        /// <summary>
        /// Gets or sets the rank of the output array. Must be greater than or equal to 1.
        /// </summary>
        [Description("Gets or sets the rank of the output array. Must be greater than or equal to 1.")]
        public int Rank { get; set; } = 1;

        /// <inheritdoc/>
        public override Expression Build(IEnumerable<Expression> arguments)
        {
            MethodInfo methodInfo = GetType().GetMethod("Process", BindingFlags.Public | BindingFlags.Instance);
            var lengths = new int[Rank];
            Type arrayType = Array.CreateInstance(Type, lengths).GetType();
            methodInfo = methodInfo.MakeGenericMethod(Type, arrayType);
            Expression sourceExpression = arguments.First();
            
            return Expression.Call(
                Expression.Constant(this),
                methodInfo,
                sourceExpression
            );
        }

        /// <summary>
        /// Converts the input tensor into an array of the specified element type.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="TResult"></typeparam>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<TResult> Process<T, TResult>(IObservable<Tensor> source) where T : unmanaged
        {
            return source.Select(tensor => (TResult)(object)tensor.data<T>().ToNDArray());
        }
    }
}