using System;
using System.Threading;
using PointProcessDecoder.Core;

namespace Bonsai.ML.PointProcessDecoder;

sealed class PointProcessModelDisposable : IDisposable
{
    private IDisposable? resource;
    public bool IsDisposed => resource == null;

    private readonly PointProcessModel model;
    public PointProcessModel Model => model;

    public PointProcessModelDisposable(PointProcessModel model, IDisposable disposable)
    {
        this.model = model ?? throw new ArgumentNullException(nameof(model));
        resource = disposable ?? throw new ArgumentNullException(nameof(disposable));
    }

    public void Dispose()
    {
        var disposable = Interlocked.Exchange(ref resource, null);
        disposable?.Dispose();
    }
}