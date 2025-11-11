using System;
using System.Reactive;
using System.Reactive.Linq;
using TorchSharp;
using Bonsai;
using System.ComponentModel;

/// <summary>
/// This operator ensures that all tensor operations within the observable sequence are executed without tracking gradients.
/// </summary>
[Combinator]
[Description("Ensures that all tensor operations within the observable sequence are executed without tracking gradients.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class ObserveWithNoGradientTracking
{
    /// <summary>
    /// Processes an observable sequence, executing all tensor operations without tracking gradients.
    /// </summary>
    public IObservable<T> Process<T>(IObservable<T> source)
    {
        return Observable.Create<T>(observer =>
        {
            var sourceObserver = Observer.Create<T>(value =>
                {
                    using var noGrad = torch.no_grad();
                    observer.OnNext(value);
                },
                observer.OnError,
                observer.OnCompleted);
            return source.SubscribeSafe(sourceObserver);
        });
    }
}