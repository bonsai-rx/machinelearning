using System;
using System.Linq;
using System.Linq.Expressions;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Reflection;
using static TorchSharp.torch;
using Bonsai.Expressions;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that saves a module's state to a file.
/// </summary>
[Combinator]
[Description("Saves a module's state to a file.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class SaveModuleStateBuilder : SingleArgumentExpressionBuilder
{
    /// <summary>
    /// The path where the module's state will be stored.
    /// </summary>
    [Description("The path where the module's state will be stored.")]
    [Editor("Bonsai.Design.SaveFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
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
            throw new InvalidOperationException("The SaveModuleState operator requires a Module type as input.");

        var selectMethod = typeof(Observable)
            .GetMethods()
            .First(m =>
                m.Name == nameof(Observable.Select) &&
                m.IsGenericMethodDefinition &&
                m.GetParameters().Length == 2);

        var moduleParameter = Expression.Parameter(inputArg, "m");

        var moduleSaveMethods = typeof(nn.Module)
            .GetMethods(BindingFlags.Instance | BindingFlags.Public)
            .Where(m =>
            {
                var parameters = m.GetParameters();
                return m.Name == "save" && parameters.Length == 2 && parameters[0].ParameterType == typeof(string) && parameters[1].ParameterType == typeof(IList<string>);
            });

        var moduleSaveMethod = moduleSaveMethods.First();

        List<Expression> moduleSaveArgs =
        [
            Expression.Constant(ModulePath, typeof(string)),
            Expression.Constant(null, typeof(IList<string>))
        ];

        var saveCall = Expression.Call(moduleParameter, moduleSaveMethod, moduleSaveArgs);

        var convertSaveCall = Expression.Convert(saveCall, inputArg);
        var lambda = Expression.Lambda(convertSaveCall, moduleParameter);

        var genericSelectMethod = selectMethod.MakeGenericMethod(inputArg, inputArg);

        return Expression.Call(genericSelectMethod, args.First(), lambda);
    }
}
