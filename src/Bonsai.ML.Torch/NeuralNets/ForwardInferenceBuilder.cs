using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Reflection;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.jit;
using System.Xml.Serialization;
using Bonsai.Expressions;
using System.Linq.Expressions;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Runs forward inference on the input tensor using the specified model.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Runs forward inference on the input tensor using the specified model.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class ForwardInferenceBuilder : SingleArgumentExpressionBuilder
{
    /// <summary>
    /// 
    /// </summary>
    /// <param name="arguments"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public override Expression Build(IEnumerable<Expression> arguments)
    {
        Console.WriteLine($"Arguments: {string.Join(", ", arguments.Select(a => a.Type.Name))}");

        var firstArg = arguments.First();
        Console.WriteLine($"First argument: {firstArg}");

        var secondArg = arguments.Last();
        Console.WriteLine($"Second argument: {secondArg}");

        var inputType = arguments.First().Type;

        Console.WriteLine($"Input type: {inputType}");

        var inputTypeName = inputType.Name;
        Console.WriteLine($"Input type name: {inputTypeName}");

        var inputTypeType = inputType.GetType();
        Console.WriteLine($"Input type type: {inputTypeType}");

        var genericInputTypes = inputType.IsGenericType ? inputType.GetGenericArguments() : [];

        Console.WriteLine($"Generic input: {genericInputTypes}. Types: {string.Join(", ", genericInputTypes.Select(t => t.Name))}. Length: {genericInputTypes.Length}");

        if (genericInputTypes.Length != 1)
            throw new InvalidOperationException("The input must be a single generic argument.");

        var genericInputType = genericInputTypes[0];
        Console.WriteLine($"Generic input type: {genericInputType}");

        // Get the generic type arguments of the tuple
        var genericArguments = genericInputType.GetGenericArguments();
        Console.WriteLine($"Generic arguments: {string.Join(", ", genericArguments.Select(t => t.Name))}");

        var sourceArguments = genericArguments.First().GetGenericArguments();

        // The last argument is the model
        var model = genericArguments.Last();
        Console.WriteLine($"Model type: {model.FullName}");
        Console.WriteLine($"Model type: {model}");

        var genericModelArguments = model.GetGenericArguments();
        Console.WriteLine($"Generic model arguments: {string.Join(", ", genericModelArguments.Select(t => t.Name))}");

        var modelOutputType = genericModelArguments.Last();
        Console.WriteLine($"Model output type: {modelOutputType.Name}");

        var modelInputTypes = genericModelArguments.Take(genericModelArguments.Length - 1).ToArray();
        Console.WriteLine($"Model input types: {string.Join(", ", modelInputTypes.Select(t => t.Name))}");

        // Ensure that the input argument types up to the model type match the model input types
        for (int i = 0; i < modelInputTypes.Length; i++)
        {
            if (sourceArguments[i] != modelInputTypes[i])
                throw new InvalidOperationException($"The input argument type '{sourceArguments[i].Name}' does not match the model input type '{modelInputTypes[i].Name}'.");
        }

        // Create a parameter expression for the model
        // var modelExpression = Expression.Call(Expression.Parameter(model, "model"), 
        // Console.WriteLine($"Model expression: {modelExpression}");

        Expression sourceParameter;
        if (modelInputTypes.Length == 1)
        {
            var observableSource = typeof(IObservable<>).MakeGenericType(modelInputTypes[0]);
            sourceParameter = Expression.Parameter(observableSource, "source");
        }
        else
        {
            // Create a tuple expression for the input types
            var tupleType = System.Type.GetType($"System.Tuple`{modelInputTypes.Length}").MakeGenericType(modelInputTypes);
            var observableSource = typeof(IObservable<>).MakeGenericType(tupleType);
            sourceParameter = Expression.Parameter(observableSource, "source");
        }

        if (sourceParameter is null)
            throw new InvalidOperationException($"The source expression could not be created: {sourceParameter}.");
        
        Console.WriteLine($"Source expression: {sourceParameter}");

        // var sourceExpression = Expression.Constant(sourceParameter, firstArg);

        // Get the appropriate Process method
        MethodInfo processMethod = null;
        if (model.IsSubclassOf(typeof(ScriptModule)))
            processMethod = typeof(ForwardInferenceBuilder).GetMethods(BindingFlags.NonPublic | BindingFlags.Static)
                .First(m => m.Name == "Process" && m.GetGenericArguments().Length == genericModelArguments.Length && m.GetParameters().Length == 2 && m.GetParameters()[1].ParameterType.IsSubclassOf(typeof(ScriptModule)))
                .MakeGenericMethod(modelInputTypes.Concat([modelOutputType]).ToArray());
        else
            processMethod = typeof(ForwardInferenceBuilder).GetMethods(BindingFlags.NonPublic | BindingFlags.Static)
                .First(m => m.Name == "Process" && m.GetGenericArguments().Length == genericModelArguments.Length && m.GetParameters().Length == 2 && m.GetParameters()[1].ParameterType.IsSubclassOf(typeof(nn.Module)))
                .MakeGenericMethod(modelInputTypes.Concat([modelOutputType]).ToArray());
        
        Console.WriteLine($"Process method: {processMethod}");
        if (processMethod is null)
            throw new InvalidOperationException($"No method overload for Process could be found that matches the input and output types: {string.Join(", ", modelInputTypes.Concat([modelOutputType]).Select(t => t.Name))}.");

        // var callExpression = Expression.Call(processMethod, sourceExpression, modelExpression);
        return Expression.Call(processMethod, firstArg, secondArg);
    }

    ///
    private static IObservable<TResult> Process<T1, TResult>(IObservable<T1> source, Module<T1, TResult> model)
    {
        return source.Select(model.forward);
    }

    private static IObservable<TResult> Process<T1, T2, TResult>(IObservable<Tuple<T1, T2>> source, Module<T1, T2, TResult> model)
    {
        return source.Select(input => model.forward(input.Item1, input.Item2));
    }

    private static IObservable<TResult> Process<T1, T2, T3, TResult>(IObservable<Tuple<T1, T2, T3>> source, Module<T1, T2, T3, TResult> model)
    {
        return source.Select(input => model.forward(input.Item1, input.Item2, input.Item3));
    }

    private static IObservable<TResult> Process<T1, T2, T3, T4, TResult>(IObservable<Tuple<T1, T2, T3, T4>> source, Module<T1, T2, T3, T4, TResult> model)
    {
        return source.Select(input => model.forward(input.Item1, input.Item2, input.Item3, input.Item4));
    }

    private static IObservable<TResult> Process<T1, T2, T3, T4, T5, TResult>(IObservable<Tuple<T1, T2, T3, T4, T5>> source, Module<T1, T2, T3, T4, T5, TResult> model)
    {
        return source.Select(input => model.forward(input.Item1, input.Item2, input.Item3, input.Item4, input.Item5));
    }

    private static IObservable<TResult> Process<T1, T2, T3, T4, T5, T6, TResult>(IObservable<Tuple<T1, T2, T3, T4, T5, T6>> source, Module<T1, T2, T3, T4, T5, T6, TResult> model)
    {
        return source.Select(input => model.forward(input.Item1, input.Item2, input.Item3, input.Item4, input.Item5, input.Item6));
    }

    private static IObservable<TResult> Process<T1, TResult>(IObservable<T1> source, ScriptModule<TResult> model)
    {
        return source.Select(input => model.forward());
    }

    private static IObservable<TResult> Process<T1, TResult>(IObservable<T1> source, ScriptModule<T1, TResult> model)
    {
        return source.Select(model.forward);
    }

    private static IObservable<TResult> Process<T1, T2, TResult>(IObservable<Tuple<T1, T2>> source, ScriptModule<T1, T2, TResult> model)
    {
        return source.Select(input => model.forward(input.Item1, input.Item2));
    }
}