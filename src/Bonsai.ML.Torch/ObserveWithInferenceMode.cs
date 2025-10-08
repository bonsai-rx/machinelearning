using System;
using System.Reactive;
using System.Reactive.Linq;
using TorchSharp;
using Bonsai;
using System.ComponentModel;

/// <summary>
/// This operator ensures that all tensor operations within the observable sequence are executed in inference mode.
/// </summary>
[Combinator]
[Description("Ensures that all tensor operations within the observable sequence are executed in inference mode.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class ObserveWithInferenceMode
{
    /// <summary>
    /// Processes an observable sequence, executing all tensor operations in inference mode.
    /// </summary>
    public IObservable<T> Process<T>(IObservable<T> source)
    {
        return Observable.Create<T>(observer =>
        {
            var sourceObserver = Observer.Create<T>(value =>
                {
                    using var inferenceMode = torch.inference_mode();
                    observer.OnNext(value);
                },
                observer.OnError,
                observer.OnCompleted);
            return source.SubscribeSafe(sourceObserver);
        });
    }
}