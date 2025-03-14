using System;
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
        private static readonly Dictionary<ScalarType, (IplDepth IplDepth, Depth Depth)> bitDepthLookup = new()
        {
            { ScalarType.Byte, (IplDepth.U8, Depth.U8) },
            { ScalarType.Int16, (IplDepth.S16, Depth.S16) },
            { ScalarType.Int32, (IplDepth.S32, Depth.S32) },
            { ScalarType.Float32, (IplDepth.F32, Depth.F32) },
            { ScalarType.Float64, (IplDepth.F64, Depth.F64) },
            { ScalarType.Int8, (IplDepth.S8, Depth.S8) }
        };

        /// <summary>
        /// Converts an OpenCV image to a Torch tensor.
        /// </summary>
        /// <param name="image"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        public static Tensor ToTensor(IplImage image, Device device)
        {
            if (image == null)
                return empty([ 0, 0, 0 ]);

            int height = image.Height;
            int channels = image.Channels;
            var width = image.WidthStep / channels;

            var iplDepth = image.Depth;
            var tensorType = bitDepthLookup.FirstOrDefault(x => x.Value.IplDepth == iplDepth).Key;

            IntPtr data = image.ImageData;
            ReadOnlySpan<long> dimensions = stackalloc long[] { height, width, channels };

            if (data == IntPtr.Zero)
                throw new InvalidOperationException($"Got {nameof(IplImage)} without backing data, this isn't expected to be possible.");

            return TorchSharpEx.CreateTensorFromUnmanagedMemoryWithManagedAnchor(data, image, dimensions, tensorType, device);
        }

        /// <summary>
        /// Converts an OpenCV mat to a Torch tensor.
        /// </summary>
        /// <param name="mat"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        public static Tensor ToTensor(Mat mat, Device device)
        {
            if (mat == null)
                return empty([0, 0, 0 ]);

            int width = mat.Size.Width;
            int height = mat.Size.Height;
            int channels = mat.Channels;

            var depth = mat.Depth;
            var tensorType = bitDepthLookup.FirstOrDefault(x => x.Value.Depth == depth).Key;

            IntPtr data = mat.Data;
            ReadOnlySpan<long> dimensions = stackalloc long[] { height, width, channels };

            if (data == IntPtr.Zero)
                throw new InvalidOperationException($"Got {nameof(Mat)} without backing data, this isn't expected to be possible.");

            return TorchSharpEx.CreateTensorFromUnmanagedMemoryWithManagedAnchor(data, mat, dimensions, tensorType, device);
        }

        private static (int height, int width, int channels) GetImageDimensions(this Tensor tensor)
        {
            if (tensor.dim() != 3)
                throw new ArgumentException("The tensor does not have exactly 3 dimensions.");

            checked
            { return ((int)tensor.size(0), (int)tensor.size(1), (int)tensor.size(2)); }
        }

        /// <summary>
        /// Converts a Torch tensor to an OpenCV image.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        public unsafe static IplImage ToImage(Tensor tensor, Device device)
        {
            var (height, width, channels) = tensor.GetImageDimensions();

            var tensorType = tensor.dtype;
            var iplDepth = bitDepthLookup[tensorType].IplDepth;
            var image = new IplImage(new OpenCV.Net.Size(width, height), iplDepth, channels);

            // Create a temporary tensor backed by the image's memory and copy the source tensor into it
            ReadOnlySpan<long> dimensions = stackalloc long[] { height, width, channels };
            using var imageTensor = TorchSharpEx.CreateStackTensor(image.ImageData, image, dimensions, tensorType, device);
            imageTensor.Tensor.copy_(tensor);

            return image;
        }

        /// <summary>
        /// Converts a Torch tensor to an OpenCV mat.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        public unsafe static Mat ToMat(Tensor tensor, Device device)
        {
            var (height, width, channels) = tensor.GetImageDimensions();

            var tensorType = tensor.dtype;
            var depth = bitDepthLookup[tensorType].Depth;
            var mat = new Mat(new OpenCV.Net.Size(width, height), depth, channels);

            // Create a temporary tensor backed by the matrix's memory and copy the source tensor into it
            ReadOnlySpan<long> dimensions = stackalloc long[] { height, width, channels };
            using var matTensor = TorchSharpEx.CreateStackTensor(mat.Data, mat, dimensions, tensorType, device);
            matTensor.Tensor.copy_(tensor);

            return mat;
        }
    }
}
