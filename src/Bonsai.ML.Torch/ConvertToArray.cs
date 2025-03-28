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
    public class ConvertToArray : SingleArgumentExpressionBuilder
    {
        private Type _type = typeof(float);
        /// <summary>
        /// Gets or sets the type of the elements in the output array.
        /// </summary>
        [Description("Gets or sets the type of the elements in the output array.")]
        [TypeConverter(typeof(ScalarTypeConverter))]
        public ScalarType Type 
        { 
            get => ScalarTypeLookup.GetScalarTypeFromType(_type);
            set => _type = ScalarTypeLookup.GetTypeFromScalarType(value);
        }

        /// <inheritdoc/>
        public override Expression Build(IEnumerable<Expression> arguments)
        {
            MethodInfo methodInfo = GetType().GetMethod("Process", BindingFlags.Public | BindingFlags.Instance);
            methodInfo = methodInfo.MakeGenericMethod(_type);
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
            return source.Select(tensor => 
            {
                if (tensor.dtype != Type)
                {
                    tensor = tensor.to_type(Type);
                }
                return tensor.data<T>().ToArray();
            });
        }
    }
}