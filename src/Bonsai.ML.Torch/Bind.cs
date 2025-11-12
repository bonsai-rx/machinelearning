using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

/// <summary>
/// Represents an operator that gathers incoming tensors into zero or more tensors by concatenating them along the first dimension.
/// </summary>
/// <remarks>
/// The operator maintains an internal buffer that accumulates incoming tensors until it reaches the specified count.
/// When the buffer reaches the specified count, it is emitted as a single tensor. After emitting the buffer, the operator skips a specified number of incoming tensors before starting to fill the buffer again.
/// </remarks>
[Combinator]
[Description("Gathers incoming tensors into zero or more tensors by concatenating them along the first dimension.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class Bind
{
    private int _count = 1;
    /// <summary>
    /// Gets or sets the number of tensors to accumulate in the buffer before emitting.
    /// </summary>
    [Description("The number of tensors to accumulate in the buffer before emitting.")]
    public int Count
    {
        get => _count;
        set => _count = value <= 0
            ? throw new ArgumentOutOfRangeException("Count must be greater than zero.")
            : value;
    }

    private int _skip = 1;
    /// <summary>
    /// Gets or sets the number of tensors to skip after emitting the buffer.
    /// </summary>
    [Description("The number of tensors to skip after emitting the buffer.")]
    public int Skip
    {
        get => _skip;
        set => _skip = value < 0
            ? throw new ArgumentOutOfRangeException("Skip must be non-negative.")
            : value;
    }

    /// <summary>
    /// Processes an observable sequence of tensors, buffering them and concatenating along the first dimension.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return Observable.Create<Tensor>(observer =>
        {
            var count = Count;
            var skip = Skip;

            Tensor buffer = null;
            int current = 0;
            Tensor idxSrc = null;
            Tensor idxDst = null;

            return source.Subscribe(input =>
            {
                if (input is null)
                    return;

                if (buffer is null)
                {
                    var shape = input.shape.Prepend(count).ToArray();
                    buffer = empty(shape, dtype: input.dtype, device: input.device);

                    if (skip < count)
                    {
                        idxSrc = arange(skip, count, dtype: ScalarType.Int32, device: input.device);
                        idxDst = arange(0, count - skip, dtype: ScalarType.Int32, device: input.device);
                    }
                }

                if (current >= 0)
                {
                    buffer[current] = input;
                }

                current++;

                if (current >= count)
                {
                    var output = buffer.clone();
                    if (skip < count)
                    {
                        var src = index_select(buffer, 0, idxSrc);
                        buffer.index_copy_(0, idxDst, src);
                        buffer[torch.TensorIndex.Slice(count - skip, null)].zero_();
                    }
                    else
                    {
                        buffer.zero_();
                    }
                    current = count - skip;
                    observer.OnNext(output);
                }
            },
            observer.OnError,
            () =>
            {
                var remainder = current;

                if (remainder > 0 && buffer is not null)
                {
                    var outputShape = buffer.shape.ToArray();
                    outputShape[0] = remainder;
                    var output = empty(outputShape, dtype: buffer.dtype, device: buffer.device);
                    for (int i = 0; i < remainder; i++)
                    {
                        output[i] = buffer[i];
                    }
                    observer.OnNext(output.clone());
                }

                buffer?.Dispose();
                idxSrc?.Dispose();
                idxDst?.Dispose();
                observer.OnCompleted();
            });
        });
    }
}