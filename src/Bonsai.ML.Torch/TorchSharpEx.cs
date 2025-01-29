#nullable enable
#pragma warning disable CS1573 // Parameter has no matching param tag in the XML comment
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp;

namespace Bonsai.ML.Torch;

internal unsafe static class TorchSharpEx
{
    [DllImport("LibTorchSharp")]
    private static extern IntPtr THSTensor_new(IntPtr rawArray, DeleterCallback deleter, long* dimensions, int numDimensions, sbyte type, sbyte dtype, int deviceType, int deviceIndex, byte requires_grad);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void DeleterCallback(IntPtr context);

    // Torch does not expect the deleter callback to be able to be null since it's a C++ reference and LibTorchSharp does not expose the functions used to create a tensor without a deleter callback, so we must use a no-op callback
    private static DeleterCallback NullDeleterCallback = _ => { };

    // Acts as GC root for unmanaged callbacks, value is unused
    private static ConcurrentDictionary<DeleterCallback, nint> ActiveDeleterCallbacks = new();

    /// <summary>Creates a <see cref="torch.Tensor"/> from unmanaged memory that is owned by a managed object</summary>
    /// <param name="data">The unmanaged memory that will back the tensor, must remain valid and fixed for the lifetime of the tensor</param>
    /// <param name="managedAnchor">The managed .NET object which owns <paramref name="data"/></param>
    public static torch.Tensor CreateTensorFromUnmanagedMemoryWithManagedAnchor(IntPtr data, object managedAnchor, ReadOnlySpan<long> dimensions, torch.ScalarType dataType)
    {
        //PERF: Ideally this would receive the GCHandle as the context rather than the pointer to the unmanaged memory since that's what we actually want to free
        // Torch itself has the ability to set the context to something else via `TensorMaker::context(void* value, ContextDeleter deleter)`, but unfortunately this method isn't exposed in LibTorchSharp
        // This is the inefficient method TorchSharp uses, which has quite a lot of unecessary overhead (particularly the unmanaged delegate allocation.)
        // Some overhead could be removed by looking up the GCHandle from the native pointer, but doing this without breaking the ability to create redundant tensors over the same data is overly complicated.
        GCHandle handle = default;
        DeleterCallback? deleter = null;
        deleter = (data) =>
        {
            if (handle.IsAllocated)
                handle.Free();

            if (!ActiveDeleterCallbacks.TryRemove(deleter!, out _))
                Debug.Fail($"The same tensor data handle deleter was called more than once!");
        };

        if (!ActiveDeleterCallbacks.TryAdd(deleter, default))
            Debug.Fail("Unreachable");

        handle = GCHandle.Alloc(managedAnchor);

        bool isInitialized = false;
        try
        {
            fixed (long* dimensionsPtr = &dimensions[0])
            {
                IntPtr tensorHandle = THSTensor_new(data, deleter, dimensionsPtr, dimensions.Length, (sbyte)dataType, (sbyte)dataType, 0, 0, 0);
                if (tensorHandle == IntPtr.Zero)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    tensorHandle = THSTensor_new(data, deleter, dimensionsPtr, dimensions.Length, (sbyte)dataType, (sbyte)dataType, 0, 0, 0);
                }

                if (tensorHandle == IntPtr.Zero)
                    torch.CheckForErrors();

                isInitialized = true;
                return torch.Tensor.UnsafeCreateTensor(tensorHandle);
            }
        }
        finally
        {
            if (!isInitialized)
                deleter(data);
        }
    }

    internal readonly ref struct StackTensor
    {
        public readonly torch.Tensor Tensor;
        private readonly object? Anchor;

        internal StackTensor(torch.Tensor tensor, object? anchor)
        {
            Tensor = tensor;
            Anchor = anchor;
        }

        public void Dispose()
        {
            Tensor.Dispose();
            GC.KeepAlive(Anchor);
        }
    }

    /// <summary>Creates a tensor which is associated with a stack scope.</summary>
    /// <param name="data">The unmanaged memory that will back the tensor, must remain valid and fixed for the lifetime of the tensor</param>
    /// <param name="managedAnchor">An optional managed .NET object which owns <paramref name="data"/></param>
    /// <remarks>
    /// The returned stack tensor must be disposed. The tensor it refers to will not be valid outside of the scope where it was allocated.
    /// </remarks>
    internal static StackTensor CreateStackTensor(IntPtr data, object? managedAnchor, ReadOnlySpan<long> dimensions, torch.ScalarType dataType)
    {
        fixed (long* dimensionsPtr = &dimensions[0])
        {
            IntPtr tensorHandle = THSTensor_new(data, NullDeleterCallback, dimensionsPtr, dimensions.Length, (sbyte)dataType, (sbyte)dataType, 0, 0, 0);
            if (tensorHandle == IntPtr.Zero)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                tensorHandle = THSTensor_new(data, NullDeleterCallback, dimensionsPtr, dimensions.Length, (sbyte)dataType, (sbyte)dataType, 0, 0, 0);
            }

            if (tensorHandle == IntPtr.Zero)
                torch.CheckForErrors();

            torch.Tensor result = torch.Tensor.UnsafeCreateTensor(tensorHandle);
            return new StackTensor(result, data);
        }
    }

    /// <summary>Gets a pointer to the tensor's backing memory</summary>
    /// <remarks>The data backing a tensor is not necessarily contiguous or even present on the CPU, consider other strategies before using this method.</remarks>
    public static IntPtr DangerousGetDataPointer(this torch.Tensor tensor)
    {
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_data(IntPtr handle);

        IntPtr data = THSTensor_data(tensor.Handle);
        if (data == IntPtr.Zero)
            torch.CheckForErrors();
        return data;
    }
}
