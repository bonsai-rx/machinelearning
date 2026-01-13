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
/// Represents an operator that saves a module's state to a file.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Saves a module's state to a file.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class SaveModule
{
    /// <summary>
    /// The path to save the module's state.
    /// </summary>
    [Description("The path to save the module's state.")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string ModulePath { get; set; }

    /// <summary>
    /// Saves the input module's state to the specified file path.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module> Process(IObservable<Module> source)
    {
        return source.Do(input =>
        {
            input.save(ModulePath);
        });
    }

    /// <summary>
    /// Saves the input module's state to the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<T1, TResult>> Process<T1, TResult>(IObservable<Module<T1, TResult>> source)
    {
        return source.Do(input =>
        {
            input.save(ModulePath);
        });
    }

    /// <summary>
    /// Saves the input module's state to the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<T1, T2, TResult>> Process<T1, T2, TResult>(IObservable<Module<T1, T2, TResult>> source)
    {
        return source.Do(input =>
        {
            input.save(ModulePath);
        });
    }

    /// <summary>
    /// Saves the input module's state to the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="T3"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<T1, T2, T3, TResult>> Process<T1, T2, T3, TResult>(IObservable<Module<T1, T2, T3, TResult>> source)
    {
        return source.Do(input =>
        {
            input.save(ModulePath);
        });
    }

    /// <summary>
    /// Saves the input module's state to the specified file path.
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
        return source.Do(input =>
        {
            input.save(ModulePath);
        });
    }

    /// <summary>
    /// Saves the input module's state to the specified file path.
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
        return source.Do(input =>
        {
            input.save(ModulePath);
        });
    }

    /// <summary>
    /// Saves the input module's state to the specified file path.
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
        return source.Do(input =>
        {
            input.save(ModulePath);
        });
    }

    /// <summary>
    /// Saves the input scripted module's state to the specified file path.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<ScriptModule> Process(IObservable<ScriptModule> source)
    {
        return source.Do(input =>
        {
            save(input, ModulePath);
        });
    }

    /// <summary>
    /// Saves the input scripted module to the specified file path.
    /// </summary>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<ScriptModule<TResult>> Process<TResult>(IObservable<ScriptModule<TResult>> source)
    {
        return source.Do(input =>
        {
            save(input, ModulePath);
        });
    }

    /// <summary>
    /// Saves the input scripted module to the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<ScriptModule<T1, TResult>> Process<T1, TResult>(IObservable<ScriptModule<T1, TResult>> source)
    {
        return source.Do(input =>
        {
            save(input, ModulePath);
        });
    }

    /// <summary>
    /// Saves the input scripted module to the specified file path.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<ScriptModule<T1, T2, TResult>> Process<T1, T2, TResult>(IObservable<ScriptModule<T1, T2, TResult>> source)
    {
        return source.Do(input =>
        {
            save(input, ModulePath);
        });
    }
}