using System;
using System.Linq;
using System.Linq.Expressions;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Reflection;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.jit;
using Bonsai.Expressions;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that loads a module's state from a file.
/// </summary>
[Combinator]
[Description("Loads a module's state from a file.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class LoadModuleStateBuilder : SingleArgumentExpressionBuilder
{
    /// <summary>
    /// The path to the module's saved state.
    /// </summary>
    [Description("The path to the module's saved state.")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string ModulePath { get; set; }

    /// <inheritdoc/>
    public override Expression Build(IEnumerable<Expression> args)
    {
        var inputArg = args.First().Type.GetGenericArguments()[0];
        bool isInputModule = false;
        for (var t = inputArg; t != null && t != typeof(object); t = t.BaseType)
        {
            if (t.GetType() == typeof(nn.Module) || t.IsGenericType && t.GetGenericTypeDefinition().Name.StartsWith("Module`"))
            {
                isInputModule = true;
                break;
            }
        }

        if (!isInputModule)
            throw new InvalidOperationException("The LoadModuleState operator requires a Module type as input.");

        var selectMethod = typeof(Observable)
            .GetMethods()
            .First(m =>
                m.Name == nameof(Observable.Select) &&
                m.IsGenericMethodDefinition &&
                m.GetParameters().Length == 2);

        var moduleParameter = Expression.Parameter(inputArg, "m");

        var moduleLoadMethods = typeof(nn.Module)
            .GetMethods(BindingFlags.Instance | BindingFlags.Public)
            .Where(m => m.Name == "load" &&
                m.GetParameters().Length == 4 &&
                m.GetParameters()[0].ParameterType == typeof(string) &&
                m.GetParameters()[1].ParameterType == typeof(bool) &&
                m.GetParameters()[2].ParameterType == typeof(IList<string>) &&
                m.GetParameters()[3].ParameterType == typeof(Dictionary<string, bool>));

        var moduleLoadMethod = moduleLoadMethods.First();

        List<Expression> moduleLoadArgs =
        [
            Expression.Constant(ModulePath, typeof(string)),
            Expression.Constant(true),                                  // strict loading
            Expression.Constant(null, typeof(IList<string>)),           // exclude list
            Expression.Constant(null, typeof(Dictionary<string, bool>)) // exclude dictionary
        ];

        var loadCall = Expression.Call(moduleParameter, moduleLoadMethod, moduleLoadArgs);

        var convertLoadCall = Expression.Convert(loadCall, inputArg);
        var lambda = Expression.Lambda(convertLoadCall, moduleParameter);

        var genericSelectMethod = selectMethod.MakeGenericMethod(inputArg, inputArg);

        return Expression.Call(genericSelectMethod, args.First(), lambda);
    }
}
