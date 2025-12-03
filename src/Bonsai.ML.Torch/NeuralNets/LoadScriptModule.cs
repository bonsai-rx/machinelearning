using System;
using System.Linq;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Linq.Expressions;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.jit;
using System.Xml.Serialization;
using Bonsai.Expressions;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Loads a TorchScript module from the specified file path.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Loads a TorchScript module from the specified file path. In order to correctly infer the module type, pass into the operator objects representing the desired ScriptModule generic argument types.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LoadScriptModule: ExpressionBuilder
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => new(0, 3);

    /// <summary>
    /// The device on which to load the model.
    /// </summary>
    [Description("The device on which to load the model.")]
    [XmlIgnore]
    public Device Device { get; set; }

    /// <summary>
    /// The path to the TorchScript module file.
    /// </summary>
    [Description("The path to the TorchScript module file.")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string ScriptModulePath { get; set; }

    /// <inheritdoc/>
    public override Expression Build(IEnumerable<Expression> arguments)
    {
        System.Type scriptModuleType;
        if (!arguments.Any())
            // if no arguments are provided, the script module type is assumed to be non-generic
            scriptModuleType = typeof(ScriptModule);
        else
        {
            // otherwise, the script module type is inferred from the collection of input argument types
            // arguments are always IObservable<T>, so we need to extract the T type from each argument
            var argumentTypes = arguments.Select(arg => arg.Type.GetGenericArguments()[0]).ToArray();
            var genericScriptModuleType = argumentTypes.Length switch
            {
                1 => typeof(ScriptModule<>),
                2 => typeof(ScriptModule<,>),
                3 => typeof(ScriptModule<,,>),
                _ => throw new NotSupportedException("Only ScriptModule types with up to three generic type arguments are supported."),
            };
            scriptModuleType = genericScriptModuleType.MakeGenericType(argumentTypes);
        }
            
        var outputType = typeof(IObservable<>).MakeGenericType(scriptModuleType);
        var modulePathExpression = Expression.Constant(ScriptModulePath, typeof(string));
        var deviceExpression = Expression.Constant(Device, typeof(Device));

        Expression callExpression;
        if (scriptModuleType.IsGenericType)
        {
            var genericArguments = scriptModuleType.GetGenericArguments();
            var processMethod = typeof(LoadScriptModule).GetMethods(System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)
                .First(m => m.Name == nameof(ProcessScriptModule) && m.IsGenericMethodDefinition && m.GetGenericArguments().Length == genericArguments.Length)
                .MakeGenericMethod(genericArguments);
            callExpression = Expression.Call(processMethod, modulePathExpression, deviceExpression);
        }
        else
        {
            var processMethod = typeof(LoadScriptModule).GetMethods(System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)
                .First(m => m.Name == nameof(ProcessScriptModule) && !m.IsGenericMethod);
            callExpression = Expression.Call(processMethod, modulePathExpression, deviceExpression);
        }

        return callExpression;
    }

    /// <summary>
    /// Loads the scripted module from the specified file path.
    /// </summary>
    /// <param name="modulePath"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    private static IObservable<ScriptModule> ProcessScriptModule(string modulePath, Device device)
    {
        var scriptModule = device is null ? jit.load(modulePath) : load(modulePath, device);
        return Observable.Return(scriptModule);
    }

    /// <summary>
    /// Loads the scripted module from the specified file path.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="modulePath"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    private static IObservable<ScriptModule<T>> ProcessScriptModule<T>(string modulePath, Device device)
    {
        var scriptModule = device is null ? load<T>(modulePath) : load<T>(modulePath, device);
        return Observable.Return(scriptModule);
    }

    /// <summary>
    /// Loads the scripted module from the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <param name="modulePath"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    private static IObservable<ScriptModule<T1, T2>> ProcessScriptModule<T1, T2>(string modulePath, Device device)
    {
        var scriptModule = device is null ? load<T1, T2>(modulePath) : load<T1, T2>(modulePath, device);
        return Observable.Return(scriptModule);
    }

    /// <summary>
    /// Loads the scripted module from the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="T3"></typeparam>
    /// <param name="modulePath"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    private static IObservable<ScriptModule<T1, T2, T3>> ProcessScriptModule<T1, T2, T3>(string modulePath, Device device)
    {
        var scriptModule = device is null ? load<T1, T2, T3>(modulePath) : load<T1, T2, T3>(modulePath, device);
        return Observable.Return(scriptModule);
    }
}