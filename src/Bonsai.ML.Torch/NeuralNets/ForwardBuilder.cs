using System;
using System.Linq;
using System.Linq.Expressions;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Reflection;
using static TorchSharp.torch;
using static TorchSharp.torch.jit;
using Bonsai.Expressions;
using TorchSharp;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that runs forward inference on the input using the specified module.
/// </summary>
[Combinator]
[Description("Runs forward inference on the input using the specified module.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class ForwardBuilder : SingleArgumentExpressionBuilder
{
    /// <inheritdoc/>
    public override Expression Build(IEnumerable<Expression> arguments)
    {
        var argumentList = arguments.First().Type.GetGenericArguments()[0];
        var inputArgs = argumentList.GetGenericArguments()[0];
        var moduleArg = argumentList.GetGenericArguments()[1];

        var sourceType = typeof(Tuple<,>).MakeGenericType(inputArgs, moduleArg);

        var selectMethod = typeof(Observable)
            .GetMethods()
            .First(m =>
                m.Name == nameof(Observable.Select) &&
                m.IsGenericMethodDefinition &&
                m.GetParameters().Length == 2);

        System.Type? moduleType = null;
        Expression? selectExpression = null;
        System.Type? resultType = null;

        for (var t = moduleArg; t != null && t != typeof(object); t = t.BaseType)
        {
            if (t.IsGenericType && t.GetGenericTypeDefinition().Name.StartsWith("Module`"))
            {
                moduleType = t;

                resultType = typeof(Tuple<,>).MakeGenericType(inputArgs, moduleType);

                var tuple1 = Expression.Parameter(sourceType, "t");
                var tuple1Item1 = Expression.Property(tuple1, "Item1");
                var tuple1Item2 = Expression.Property(tuple1, "Item2");

                var moduleConversionExpression = Expression.Convert(tuple1Item2, moduleType);
                var tupleCreateMethod = typeof(Tuple)
                    .GetMethods()
                    .Single(m =>
                        m.Name == nameof(Tuple.Create) &&
                        m.IsGenericMethodDefinition &&
                        m.GetGenericArguments().Length == 2 &&
                        m.GetParameters().Length == 2)
                    .MakeGenericMethod(inputArgs, moduleType);

                var newTuple = Expression.Call(tupleCreateMethod, tuple1Item1, moduleConversionExpression);

                var selector = Expression.Lambda(newTuple, tuple1);

                selectExpression = Expression.Call(selectMethod.MakeGenericMethod(sourceType, resultType), arguments.First(), selector);
                break;
            }

            else if (t.IsGenericType && t.GetGenericTypeDefinition().Name.StartsWith("ScriptModule`"))
            {
                moduleType = t;
                resultType = sourceType;
                selectExpression = arguments.First();
                break;
            }
        }

        if (moduleType is null)
            throw new InvalidOperationException("The specified module type is not a valid TorchSharp module.");

        var tuple = Expression.Parameter(resultType, "t");
        var item1 = Expression.Property(tuple, "Item1");
        var item2 = Expression.Property(tuple, "Item2");

        List<Expression> forwardCallArgs = [];

        if (inputArgs.IsGenericType && inputArgs.GetGenericTypeDefinition().Name.StartsWith("Tuple`"))
        {
            var inputArgsTypes = inputArgs.GetGenericArguments();
            for (int i = 0; i < inputArgsTypes.Length; i++)
            {
                var itemN = Expression.Property(item1, $"Item{i + 1}");
                forwardCallArgs.Add(itemN);
            }
        }
        else
        {
            forwardCallArgs.Add(item1);
        }

        var moduleMethods = moduleType
            .GetMethods(BindingFlags.Instance | BindingFlags.Public)
            .Where(m => m.Name == "forward" &&
                m.IsPublic &&
                m.GetParameters().Length == forwardCallArgs.Count &&
                m.GetParameters().Select(p => p.ParameterType).SequenceEqual(forwardCallArgs.Select(a => a.Type)));

        if (!moduleMethods.Any())
            throw new InvalidOperationException("The module does not contain a matching forward method.");

        var forwardMethod = moduleMethods.First();

        var forwardCall = Expression.Call(item2, forwardMethod, forwardCallArgs);
        var forwardLambda = Expression.Lambda(forwardCall, tuple);

        return Expression.Call(selectMethod.MakeGenericMethod(resultType, forwardMethod.ReturnType), selectExpression, forwardLambda);
    }
}
