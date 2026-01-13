using System;
using System.Linq;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.jit;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that runs forward inference on the input using the specified module.
/// </summary>
[Combinator]
[Description("Runs forward inference on the input using the specified module.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Forward
{
    /// <summary>
    /// Runs forward inference on the input using the specified module.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TResult> Process<T1, TResult>(IObservable<Tuple<T1, Module<T1, TResult>>> source)
    {
        return source.Select(input => input.Item2.forward(input.Item1));
    }

    /// <summary>
    /// Runs forward inference on the input using the specified module.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TResult> Process<T1, T2, TResult>(IObservable<Tuple<Tuple<T1, T2>, Module<T1, T2, TResult>>> source)
    {
        return source.Select(input => input.Item2.forward(input.Item1.Item1, input.Item1.Item2));
    }

    /// <summary>
    /// Runs forward inference on the input using the specified module.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="T3"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TResult> Process<T1, T2, T3, TResult>(IObservable<Tuple<Tuple<T1, T2, T3>, Module<T1, T2, T3, TResult>>> source)
    {
        return source.Select(input => input.Item2.forward(input.Item1.Item1, input.Item1.Item2, input.Item1.Item3));
    }

    /// <summary>
    /// Runs forward inference on the input using the specified module.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="T3"></typeparam>
    /// <typeparam name="T4"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TResult> Process<T1, T2, T3, T4, TResult>(IObservable<Tuple<Tuple<T1, T2, T3, T4>, Module<T1, T2, T3, T4, TResult>>> source)
    {
        return source.Select(input => input.Item2.forward(input.Item1.Item1, input.Item1.Item2, input.Item1.Item3, input.Item1.Item4));
    }

    /// <summary>
    /// Runs forward inference on the input using the specified module.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="T3"></typeparam>
    /// <typeparam name="T4"></typeparam>
    /// <typeparam name="T5"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TResult> Process<T1, T2, T3, T4, T5, TResult>(IObservable<Tuple<Tuple<T1, T2, T3, T4, T5>, Module<T1, T2, T3, T4, T5, TResult>>> source)
    {
        return source.Select(input => input.Item2.forward(input.Item1.Item1, input.Item1.Item2, input.Item1.Item3, input.Item1.Item4, input.Item1.Item5));
    }

    /// <summary>
    /// Runs forward inference on the input using the specified module.
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
    public IObservable<TResult> Process<T1, T2, T3, T4, T5, T6, TResult>(IObservable<Tuple<Tuple<T1, T2, T3, T4, T5, T6>, Module<T1, T2, T3, T4, T5, T6, TResult>>> source)
    {
        return source.Select(input => input.Item2.forward(input.Item1.Item1, input.Item1.Item2, input.Item1.Item3, input.Item1.Item4, input.Item1.Item5, input.Item1.Item6));
    }

    /// <summary>
    /// Runs forward inference on the input using the specified module.
    /// </summary>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TResult> Process<TResult>(IObservable<ScriptModule<TResult>> source)
    {
        return source.Select(input => input.forward());
    }

    /// <summary>
    /// Runs forward inference on the input using the specified module.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TResult> Process<T1, TResult>(IObservable<Tuple<T1, ScriptModule<T1, TResult>>> source)
    {
        return source.Select(input => input.Item2.forward(input.Item1));
    }

    /// <summary>
    /// Runs forward inference on the input using the specified module.
    /// </summary>
    /// <typeparam name="T1"></typeparam>
    /// <typeparam name="T2"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TResult> Process<T1, T2, TResult>(IObservable<Tuple<Tuple<T1, T2>, ScriptModule<T1, T2, TResult>>> source)
    {
        return source.Select(input => input.Item2.forward(input.Item1.Item1, input.Item1.Item2));
    }
}