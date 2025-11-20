using System;
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
[XmlInclude(typeof(TypeMapping<ScriptModule>))]
[XmlInclude(typeof(TypeMapping<ScriptModule<Tensor>>))]
[XmlInclude(typeof(TypeMapping<ScriptModule<Tensor, Tensor>>))]
[Combinator]
[ResetCombinator]
[Description("Loads a TorchScript module from the specified file path.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LoadScriptModule: ExpressionBuilder
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => new(0, 1);

    /// <summary>
    /// The device on which to load the model.
    /// </summary>
    [Description("The device on which to load the model.")]
    [XmlIgnore]
    public Device Device { get; set; }

    /// <summary>
    /// The path to the TorchScript model file.
    /// </summary>
    [Description("The path to the TorchScript model file.")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string ModelPath { get; set; }

    /// <summary>
    /// The type mapping for the loaded script module.
    /// </summary>
    public TypeMapping ScriptModuleType { get; set; } = new TypeMapping<ScriptModule<Tensor, Tensor>>();

    /// <inheritdoc/>
    public override Expression Build(IEnumerable<Expression> arguments)
    {
        var scriptModuleType = ScriptModuleType.GetType().GetGenericArguments()[0];
        var outputType = typeof(IObservable<>).MakeGenericType(scriptModuleType);
        var modelPathExpression = Expression.Constant(ModelPath, typeof(string));
        var deviceExpression = Expression.Constant(Device, typeof(Device));

        var callExpression = scriptModuleType switch
        {
            System.Type t when t == typeof(ScriptModule) =>
                Expression.Call(typeof(LoadScriptModule).GetMethod(nameof(ProcessScriptModule), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static), modelPathExpression, deviceExpression),
            System.Type t when t == typeof(ScriptModule<Tensor>) =>
                Expression.Call(typeof(LoadScriptModule).GetMethod(nameof(ProcessScriptModuleTensor), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static), modelPathExpression, deviceExpression),
            System.Type t when t == typeof(ScriptModule<Tensor, Tensor>) =>
                Expression.Call(typeof(LoadScriptModule).GetMethod(nameof(ProcessScriptModuleTensorTensor), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static), modelPathExpression, deviceExpression),
            _ => throw new NotSupportedException($"The specified script module type '{scriptModuleType}' is not supported."),
        };

        return Expression.Call(callExpression.Method, callExpression.Arguments);
    }

    private static IObservable<ScriptModule> ProcessScriptModule(string modelPath, Device device)
    {
        var scriptModule = device is null ? jit.load(modelPath) : load(modelPath, device);
        return Observable.Return(scriptModule);
    }

    private static IObservable<ScriptModule<Tensor>> ProcessScriptModuleTensor(string modelPath, Device device)
    {
        var scriptModule = device is null ? load<Tensor>(modelPath) : load<Tensor>(modelPath, device);
        return Observable.Return(scriptModule);
    }

        private static IObservable<ScriptModule<Tensor, Tensor>> ProcessScriptModuleTensorTensor(string modelPath, Device device)
    {
        var scriptModule = device is null ? load<Tensor, Tensor>(modelPath) : load<Tensor, Tensor>(modelPath, device);
        return Observable.Return(scriptModule);
    }
}