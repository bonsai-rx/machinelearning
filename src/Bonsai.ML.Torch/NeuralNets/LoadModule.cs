using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.jit;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that loads a module's state from a file.
/// </summary>
[Combinator]
[Description("Loads a module's state from a file.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class LoadModule
{
    /// <summary>
    /// The path to the module's saved state.
    /// </summary>
    [Description("The path to the module's saved state.")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string ModulePath { get; set; }

    /// <summary>
    /// Loads the module's state from the specified file path.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module> Process(IObservable<Module> source)
    {
        return source.Select(module =>
        {
            return module.load(ModulePath);
        });
    }

    /// <summary>
    /// Loads the module's state from the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<T1, TResult>> Process<T1, TResult>(IObservable<Module<T1, TResult>> source)
    {
        return source.Select(module =>
        {
            return (Module<T1, TResult>)module.load(ModulePath);
        });
    }

    /// <summary>
    /// Loads the module's state from the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<T1, T2, TResult>> Process<T1, T2, TResult>(IObservable<Module<T1, T2, TResult>> source)
    {
        return source.Select(module =>
        {
            return (Module<T1, T2, TResult>)module.load(ModulePath);
        });
    }

    /// <summary>
    /// Loads the module's state from the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="T3"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<T1, T2, T3, TResult>> Process<T1, T2, T3, TResult>(IObservable<Module<T1, T2, T3, TResult>> source)
    {
        return source.Select(module =>
        {
            return (Module<T1, T2, T3, TResult>)module.load(ModulePath);
        });
    }

    /// <summary>
    /// Loads the module's state from the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="T3"></typeparam>
    /// <typeparam name="T4"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<T1, T2, T3, T4, TResult>> Process<T1, T2, T3, T4, TResult>(IObservable<Module<T1, T2, T3, T4, TResult>> source)
    {
        return source.Select(module =>
        {
            return (Module<T1, T2, T3, T4, TResult>)module.load(ModulePath);
        });
    }

    /// <summary>
    /// Loads the module's state from the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="T3"></typeparam>
    /// <typeparam name="T4"></typeparam>
    /// <typeparam name="T5"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<T1, T2, T3, T4, T5, TResult>> Process<T1, T2, T3, T4, T5, TResult>(IObservable<Module<T1, T2, T3, T4, T5, TResult>> source)
    {
        return source.Select(module =>
        {
            return (Module<T1, T2, T3, T4, T5, TResult>)module.load(ModulePath);
        });
    }

    /// <summary>
    /// Loads the module's state from the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="T3"></typeparam>
    /// <typeparam name="T4"></typeparam>
    /// <typeparam name="T5"></typeparam>
    /// <typeparam name="T6"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<T1, T2, T3, T4, T5, T6, TResult>> Process<T1, T2, T3, T4, T5, T6, TResult>(IObservable<Module<T1, T2, T3, T4, T5, T6, TResult>> source)
    {
        return source.Select(module =>
        {
            return (Module<T1, T2, T3, T4, T5, T6, TResult>)module.load(ModulePath);
        });
    }

    /// <summary>
    /// Loads the module's state from the specified file path.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<ScriptModule> Process(IObservable<ScriptModule> source)
    {
        return source.Select(module =>
        {
            return (ScriptModule)module.load(ModulePath);
        });
    }

    /// <summary>
    /// Loads the module's state from the specified file path.
    /// </summary>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<ScriptModule<TResult>> Process<TResult>(IObservable<ScriptModule<TResult>> source)
    {
        return source.Select(module =>
        {
            return (ScriptModule<TResult>)module.load(ModulePath);
        });
    }

    /// <summary>
    /// Loads the module's state from the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<ScriptModule<T1, TResult>> Process<T1, TResult>(IObservable<ScriptModule<T1, TResult>> source)
    {
        return source.Select(module =>
        {
            return (ScriptModule<T1, TResult>)module.load(ModulePath);
        });
    }

    /// <summary>
    /// Loads the module's state from the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<ScriptModule<T1, T2, TResult>> Process<T1, T2, TResult>(IObservable<ScriptModule<T1, T2, TResult>> source)
    {
        return source.Select(module =>
        {
            return (ScriptModule<T1, T2, TResult>)module.load(ModulePath);
        });
    }
}