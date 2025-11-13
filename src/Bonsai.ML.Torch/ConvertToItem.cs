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

namespace Bonsai.ML.Torch;

/// <summary>
/// Represents an operator that converts the input tensor into a single value of the specified element type. The input tensor must only contain a single element.
/// </summary>
[Combinator]
[Description("Converts the input tensor into a single value of the specified element type.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class ConvertToItem : SingleArgumentExpressionBuilder
{
    /// <summary>
    /// Gets or sets the type of the item.
    /// </summary>
    [Description("Gets or sets the type of the item.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <inheritdoc/>
    public override Expression Build(IEnumerable<Expression> arguments)
    {
        MethodInfo methodInfo = GetType().GetMethod("Process", BindingFlags.Public | BindingFlags.Instance);
        var type = ScalarTypeLookup.GetTypeFromScalarType(Type);
        methodInfo = methodInfo.MakeGenericMethod(type);
        Expression sourceExpression = arguments.First();
        
        return Expression.Call(
            Expression.Constant(this),
            methodInfo,
            sourceExpression
        );
    }

    /// <summary>
    /// Converts the input tensor into a single item.
    /// </summary>
    /// <returns></returns>
    public IObservable<T> Process<T>(IObservable<Tensor> source) where T : unmanaged
    {
        return source.Select(tensor => 
        {
            if (tensor.dtype != Type)
            {
                tensor = tensor.to_type(Type);
            }
            return tensor.item<T>();
        });
    }
}