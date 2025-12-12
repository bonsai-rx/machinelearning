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
    public class ConvertToNDArray : SingleArgumentExpressionBuilder
    {
        /// <summary>
        /// Gets or sets the type of the elements in the output array.
        /// </summary>
        [Description("Gets or sets the type of the elements in the output array.")]
        [TypeConverter(typeof(ScalarTypeConverter))]
        public ScalarType Type { get; set; } = ScalarType.Float32;

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
            var type = ScalarTypeLookup.GetTypeFromScalarType(Type);
            Type arrayType = Array.CreateInstance(type, lengths).GetType();
            methodInfo = methodInfo.MakeGenericMethod(type, arrayType);
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
        /// <typeparam name="T">The element type of the output item.</typeparam>
        /// <typeparam name="TResult">The type of the output array.</typeparam>
        /// <param name="source">The sequence of input tensors.</param>
        /// <returns>The sequence of output arrays of the specified element type and rank.</returns>
        public IObservable<TResult> Process<T, TResult>(IObservable<Tensor> source) where T : unmanaged
        {
            return source.Select(tensor =>
            {
                if (tensor.dtype != Type)
                {
                    tensor = tensor.to_type(Type);
                }
                return (TResult)(object)tensor.data<T>().ToNDArray();
            });
        }
    }
}
