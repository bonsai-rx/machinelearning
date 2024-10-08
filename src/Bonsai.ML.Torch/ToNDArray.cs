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
using TorchSharp.Utils;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Converts the input tensor into an array of the specified element type.
    /// </summary>
    [Combinator]
    [Description("Converts the input tensor into an array of the specified element type.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    [XmlInclude(typeof(TypeMapping<byte>))]
    [XmlInclude(typeof(TypeMapping<sbyte>))]
    [XmlInclude(typeof(TypeMapping<short>))]
    [XmlInclude(typeof(TypeMapping<int>))]
    [XmlInclude(typeof(TypeMapping<long>))]
    [XmlInclude(typeof(TypeMapping<float>))]
    [XmlInclude(typeof(TypeMapping<double>))]
    [XmlInclude(typeof(TypeMapping<bool>))]
    public class ToNDArray : SingleArgumentExpressionBuilder
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ToNDArray"/> class.
        /// </summary>
        public ToNDArray()
        {
            Type = new TypeMapping<double>();
        }

        /// <summary>
        /// Gets or sets the type mapping used to convert the input tensor into an array.
        /// </summary>
        public TypeMapping Type { get; set; }
        public int Rank { get; set; }

        /// <inheritdoc/>
        public override Expression Build(IEnumerable<Expression> arguments)
        {
            TypeMapping typeMapping = Type;
            var returnType = typeMapping.GetType().GetGenericArguments()[0];
            MethodInfo methodInfo = GetType().GetMethod("Process", BindingFlags.Public | BindingFlags.Instance);
            var lengths = new int[Rank];
            Type arrayType = Array.CreateInstance(returnType, lengths).GetType();
            methodInfo = methodInfo.MakeGenericMethod(returnType, arrayType);
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
            return source.Select(tensor =>
            {
                return (TResult)(object)tensor.data<T>().ToNDArray();
            });
        }
    }
}