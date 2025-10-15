using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;

namespace Bonsai.ML.Torch;

/// <summary>
/// This operator collects incoming tensors into a buffer and concatenates them along the first dimension.
/// </summary>
/// <remarks>
/// The operator maintains an internal buffer that accumulates incoming tensors until it reaches the specified count.
/// When the buffer reaches the specified count, it is emitted as a single tensor. After emitting the buffer, the operator skips a specified number of incoming tensors before starting to fill the buffer again.
/// </remarks>
[Combinator]
[Description("Buffers the incoming tensors and concatenates them into a single tensor along the first dimension.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class Buffer
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

    private torch.Tensor _buffer = null;
    private int _current = 0;
    private torch.Tensor _idxSrc = null;
    private torch.Tensor _idxDst = null;

    /// <summary>
    /// Processes an observable sequence of tensors, buffering them and concatenating along the first dimension.
    /// </summary>
    public IObservable<torch.Tensor> Process(IObservable<torch.Tensor> source)
    {
        var count = _count;
        var skip = _skip;
        var send = false;
        _current = 0;
        return source.Select((input) =>
        {
            if (input is null) return false;

            if (_buffer is null)
            {
                var shape = input.shape.Prepend(count).ToArray();
                _buffer = torch.empty(shape, dtype: input.dtype, device: input.device);

                if (skip < count)
                {
                    _idxSrc = torch.arange(skip, count, dtype: torch.ScalarType.Int64, device: input.device);
                    _idxDst = torch.arange(0, count - skip, dtype: torch.ScalarType.Int64, device: input.device);
                }
            }

            if (_current >= 0)
            {
                _buffer[_current] = input;
            }

            _current++;

            if (_current >= count)
            {
                send = true;
            }
            return send;
        })
        .Where(x => x)
        .Select(x =>
        {
            var output = _buffer.clone();
            if (skip < count)
            {
                var src = torch.index_select(_buffer, 0, _idxSrc);
                _buffer.index_copy_(0, _idxDst, src);
                _buffer[torch.TensorIndex.Slice(count - skip, null)].zero_();
            }
            else
            {
                _buffer.zero_();
            }
            _current = count - skip;
            send = false;
            return output;
        })
        .Finally(() =>
        {
            _buffer?.Dispose();
            _buffer = null;
            _idxSrc?.Dispose();
            _idxSrc = null;
            _idxDst?.Dispose();
            _idxDst = null;
            send = false;
        });
    }
}