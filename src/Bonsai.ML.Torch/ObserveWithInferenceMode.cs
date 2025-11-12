using System;
using System.Reactive;
using System.Reactive.Linq;
using TorchSharp;
using Bonsai;
using System.ComponentModel;

/// <summary>
/// Represents an operator that ensures all tensor operations within the observable sequence are executed in inference mode.
/// </summary>
[Combinator]
[Description("Ensures all tensor operations within the observable sequence are executed in inference mode.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class ObserveWithInferenceMode
{
    /// <summary>
    /// Returns an observable sequence which is identical to the source sequence, but where all tensor operations are executed in inference mode.
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