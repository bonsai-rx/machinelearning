using System;
using System.Runtime.InteropServices;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using OpenCV.Net;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Helper class to convert between OpenCV mats, images and Torch tensors.
    /// </summary>
    public static class OpenCVHelper
    {
        private static Dictionary<ScalarType, (IplDepth IplDepth, Depth Depth)> bitDepthLookup = new Dictionary<ScalarType, (IplDepth, Depth)> 
        {
            { ScalarType.Byte, (IplDepth.U8, Depth.U8) },
            { ScalarType.Int16, (IplDepth.S16, Depth.S16) },
            { ScalarType.Int32, (IplDepth.S32, Depth.S32) },
            { ScalarType.Float32, (IplDepth.F32, Depth.F32) },
            { ScalarType.Float64, (IplDepth.F64, Depth.F64) },
            { ScalarType.Int8, (IplDepth.S8, Depth.S8) }
        };

        private static ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();

        internal delegate void GCHandleDeleter(IntPtr memory);
        
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_data(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, IntPtr dimensions, int numDimensions, sbyte type, sbyte dtype, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        /// <summary>
        /// Creates a tensor from a pointer to the data and the dimensions of the tensor.
        /// </summary>
        /// <param name="tensorDataPtr"></param>
        /// <param name="dimensions"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        public static unsafe Tensor CreateTensorFromPtr(IntPtr tensorDataPtr, long[] dimensions, ScalarType dtype = ScalarType.Byte)
        {
            var dataHandle = GCHandle.Alloc(tensorDataPtr, GCHandleType.Pinned);
            var gchp = GCHandle.ToIntPtr(dataHandle);
            GCHandleDeleter deleter = null;

            deleter = new GCHandleDeleter((IntPtr ptrHandler) =>
            {
                GCHandle.FromIntPtr(gchp).Free();
                deleters.TryRemove(deleter, out deleter);
            });
            deleters.TryAdd(deleter, deleter);

            fixed (long* dimensionsPtr = dimensions)
            {
                IntPtr tensorHandle = THSTensor_new(tensorDataPtr, deleter, (IntPtr)dimensionsPtr, dimensions.Length, (sbyte)dtype, (sbyte)dtype, 0, 0, false);
                if (tensorHandle == IntPtr.Zero) 
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    tensorHandle = THSTensor_new(tensorDataPtr, deleter, (IntPtr)dimensionsPtr, dimensions.Length, (sbyte)dtype, (sbyte)dtype, 0, 0, false);
                }
                if (tensorHandle == IntPtr.Zero) 
                {
                    CheckForErrors(); 
                }
                var output = Tensor.UnsafeCreateTensor(tensorHandle);
                return output;
            }
        }

        /// <summary>
        /// Converts an OpenCV image to a Torch tensor.
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        public static Tensor ToTensor(IplImage image)
        {
            if (image == null)
            {
                return empty([ 0, 0, 0 ]);
            }

            int width = image.Width;
            int height = image.Height;
            int channels = image.Channels;

            var iplDepth = image.Depth;
            var tensorType = bitDepthLookup.FirstOrDefault(x => x.Value.IplDepth == iplDepth).Key;

            IntPtr tensorDataPtr = image.ImageData;
            long[] dimensions = [ height, width, channels ];
            if (tensorDataPtr == IntPtr.Zero) 
            {
                return empty(dimensions);
            }
            return CreateTensorFromPtr(tensorDataPtr, dimensions, tensorType);
        }

        /// <summary>
        /// Converts an OpenCV mat to a Torch tensor.
        /// </summary>
        /// <param name="mat"></param>
        /// <returns></returns>
        public static Tensor ToTensor(Mat mat)
        {
            if (mat == null)
            {
                return empty([0, 0, 0 ]);
            }

            int width = mat.Size.Width;
            int height = mat.Size.Height;
            int channels = mat.Channels;

            var depth = mat.Depth;
            var tensorType = bitDepthLookup.FirstOrDefault(x => x.Value.Depth == depth).Key;

            IntPtr tensorDataPtr = mat.Data;
            long[] dimensions = [ height, width, channels ];
            if (tensorDataPtr == IntPtr.Zero) 
            {
                return empty(dimensions);
            }
            return CreateTensorFromPtr(tensorDataPtr, dimensions, tensorType);  
        }

        /// <summary>
        /// Converts a Torch tensor to an OpenCV image.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public unsafe static IplImage ToImage(Tensor tensor)
        {
            var height = (int)tensor.shape[0];
            var width = (int)tensor.shape[1];
            var channels = (int)tensor.shape[2];

            var tensorType = tensor.dtype;
            var iplDepth = bitDepthLookup[tensorType].IplDepth;

            var new_tensor = zeros(new long[] { height, width, channels }, tensorType).copy_(tensor);

            var res = THSTensor_data(new_tensor.Handle);
            var image = new IplImage(new OpenCV.Net.Size(width, height), iplDepth, channels, res);

            return image;
        }

        /// <summary>
        /// Converts a Torch tensor to an OpenCV mat.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public unsafe static Mat ToMat(Tensor tensor)
        {
            var height = (int)tensor.shape[0];
            var width = (int)tensor.shape[1];
            var channels = (int)tensor.shape[2];

            var tensorType = tensor.dtype;
            var depth = bitDepthLookup[tensorType].Depth;

            var new_tensor = zeros(new long[] { height, width, channels }, tensorType).copy_(tensor);

            var res = THSTensor_data(new_tensor.Handle);
            var mat = new Mat(new OpenCV.Net.Size(width, height), depth, channels, res);

            return mat;
        }
    }
}