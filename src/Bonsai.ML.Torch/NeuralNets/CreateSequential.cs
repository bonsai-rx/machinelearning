using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Xml.Serialization;
using System.Linq;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Creates a sequential model from the specified modules.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a sequential model from the specified modules.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CreateSequential
{
    /// <summary>
    /// The modules to include in the sequential model.
    /// </summary>
    [XmlIgnore]
    public IModule<Tensor, Tensor>[] Modules { get; set; }

    /// <summary>
    /// The device on which to create the sequential model.
    /// </summary>
    [XmlIgnore]
    public Device? Device { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a sequential model from the specified modules.
    /// </summary>
    /// <returns></returns>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Defer(() =>
        {
            var sequential = Sequential([.. Modules.Select(m => (Module<Tensor, Tensor>)m)]);
            if (Device is not null && Device != CPU)
            {
                sequential.to(Device);
            }
            return Observable.Return(sequential);
        });
    }
    
    /// <summary>
    /// Generates an observable sequence that creates a sequential model from the input modules.
    /// </summary>
    /// <returns></returns>
    public IObservable<IModule<Tensor, Tensor>> Process(IObservable<IModule<Tensor, Tensor>[]> source)
    {
        return source.SelectMany(modules =>
        {
            var sequential = Sequential([.. modules.Select(m => (Module<Tensor, Tensor>)m)]);
            if (Device is not null && Device != CPU)
            {
                sequential.to(Device);
            }
            return Observable.Return(sequential);
        });
    }

    /// <summary>
    /// Generates an observable sequence of sequential models for each element in the input sequence.
    /// </summary>
    /// <returns></returns>
    public IObservable<IModule<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.SelectMany(_ =>
        {
            var sequential = Sequential([..Modules.Select(m => (Module<Tensor, Tensor>)m)]);
            if (Device is not null && Device != CPU)
            {
                sequential.to(Device);
            }
            return Observable.Return(sequential);
        });
    }
}