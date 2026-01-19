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
[Description("Loads a TorchScript module from the specified file path. In order to correctly infer the module type, pass into the operator objects representing the desired ScriptModule generic argument types.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LoadScriptModule
{
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

    /// <summary>
    /// Loads the scripted module from the specified file path.
    /// </summary>
    /// <returns></returns>
    public IObservable<ScriptModule<Tensor, Tensor>> Process()
    {
        return Observable.Defer(() =>
        {
            var scriptModule = Device is null ? load<Tensor, Tensor>(ScriptModulePath) : load<Tensor, Tensor>(ScriptModulePath, Device);
            return Observable.Return(scriptModule);
        });
    }
}
