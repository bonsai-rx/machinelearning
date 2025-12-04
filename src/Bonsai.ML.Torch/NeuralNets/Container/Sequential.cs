using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Xml.Serialization;
using System.Linq;

namespace Bonsai.ML.Torch.NeuralNets.Container;

/// <summary>
/// Creates a sequential model from the specified modules.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a sequential model from the specified modules.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Sequential
{
    /// <summary>
    /// The device on which to create the sequential model.
    /// </summary>
    [XmlIgnore]
    public Device? Device { get; set; } = null;
    
    /// <summary>
    /// Generates an observable sequence that creates a sequential model from the input modules.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process(IObservable<Module<Tensor, Tensor>[]> source)
    {
        return source.SelectMany(modules =>
        {
            var sequential = Sequential(modules);
            if (Device is not null && Device != CPU)
            {
                sequential.to(Device);
            }
            return Observable.Return(sequential);
        });
    }
}