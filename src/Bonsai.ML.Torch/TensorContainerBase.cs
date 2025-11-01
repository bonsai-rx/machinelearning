using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.CompilerServices;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

/// <summary>
/// Base class for operators that contain tensor properties. Provides automatic scalar type conversion on registered tensors.
/// </summary>
public abstract class TensorContainerBase : IScalarTypeProvider
{
    private ScalarType _scalarType = ScalarType.Float32;
    private readonly List<RegisteredTensor> _registeredTensors = new();
    private readonly object _sync = new();

    /// <inheritdoc/>
    public ScalarType Type
    {
        get => _scalarType;
        set
        {
            if (_scalarType == value) return;
            _scalarType = value;
            ConvertAllTensors();
        }
    }

    /// <summary>
    /// Registers a tensor property for automatic scalar type conversion.
    /// </summary>
    /// <param name="getter"></param>
    /// <param name="setter"></param>
    protected void RegisterTensor(Func<Tensor?> getter, Action<Tensor?> setter)
    {
        _registeredTensors.Add(new RegisteredTensor(getter, setter));
    }

    /// <summary>
    /// Convert a single tensor to the current scalar type.
    /// </summary>
    protected Tensor ConvertTensor(Tensor value)
    {
        return value.dtype == _scalarType ? value : value.to_type(_scalarType);
    }

    /// <summary>
    /// Converts all tensor properties to the current scalar type.
    /// </summary>
    protected void ConvertAllTensors()
    {
        lock (_sync)
        {
            foreach (var registeredTensor in _registeredTensors)
            {
                var tensor = registeredTensor.Getter();
                if (tensor is null || tensor.dtype == _scalarType) continue;
                registeredTensor.Setter(tensor.to_type(_scalarType));
            }
        }
    }

    private readonly struct RegisteredTensor(
        Func<Tensor> getter,
        Action<Tensor> setter)
    {
        public Func<Tensor> Getter { get; } = getter;
        public Action<Tensor> Setter { get; } = setter;
    }
}
