using System;
using System.Threading;
using PointProcessDecoder.Core;

namespace Bonsai.ML.PointProcessDecoder;

internal sealed class PointProcessModelDisposable(PointProcessModel model, IDisposable disposable) : IDisposable
{
    private IDisposable? resource = disposable ?? throw new ArgumentNullException(nameof(disposable));
    /// <summary>
    /// Gets a value indicating whether the object has been disposed.
    /// </summary>
    public bool IsDisposed => resource == null;

    private readonly PointProcessModel model = model ?? throw new ArgumentNullException(nameof(model));
    /// <summary>
    /// Gets the point process model.
    /// </summary>
    public PointProcessModel Model => model;

    public void Dispose()
    {
        var disposable = Interlocked.Exchange(ref resource, null);
        disposable?.Dispose();
    }
}