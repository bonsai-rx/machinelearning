using System;
using System.Collections.Generic;
using OpenCV.Net;
using TorchSharp;
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

        private static ScalarType GetTensorType(IplDepth depth)
        {
            foreach (var entry in bitDepthLookup)
            {
                if (entry.Value.IplDepth == depth)
                    return entry.Key;
            }

            throw new NotSupportedException($"Images with a depth of {depth} cannot be converted to a tensor.");
        }

        private static ScalarType GetTensorType(Depth depth)
        {
            foreach (var entry in bitDepthLookup)
            {
                if (entry.Value.Depth == depth)
                    return entry.Key;
            }

            throw new NotSupportedException($"Matrices with a depth of {depth} cannot be converted to a tensor.");
        }

        private static (IplDepth IplDepth, Depth Depth) GetDepth(ScalarType tensorType)
        {
            if (bitDepthLookup.TryGetValue(tensorType, out var depth))
                return depth;

            throw new NotSupportedException($"Tensors of type {tensorType} cannot be converted to an image or a matrix.");
        }

        // Describes how the pixels of an image or matrix are laid out in its buffer, in units of elements.
        // Rows are separated by the row step of the buffer, which is larger than the length of a row when
        // rows are padded for alignment or when the buffer is a region of a larger image or matrix.
        private static (long Length, long[] Size, long[] Strides) GetLayout(int height, int width, int channels, int step, ScalarType tensorType)
        {
            long rowLength = (long)width * channels;
            long rowStride = rowLength;

            // The step of a single row is never used to move between rows and can be reported as zero.
            if (height > 1)
            {
                var elementSize = tensorType.ElementSize();
                // Enforce that the row step is a multiple of the element size. This can happen if the caller supplies a custom step size over their own buffer. OpenCV does not require that the step be a multiple of the element size, but a strided tensor expects it to be.
                if (step % elementSize != 0)
                    throw new NotSupportedException($"A row step of {step} bytes is not a multiple of the element size of {elementSize} bytes.");

                rowStride = step / elementSize;
            }

            // The buffer ends at the last element of the last row, so that no memory after it is described.
            long length = (height - 1) * rowStride + rowLength;
            return (length, [height, width, channels], [rowStride, channels, 1]);
        }

        private static Tensor CreateTensor(IntPtr data, object anchor, int height, int width, int channels, int step, ScalarType tensorType, Device device)
        {
            if (height == 0 || width == 0)
                return empty([height, width, channels], dtype: tensorType, device: device);

            var (length, size, strides) = GetLayout(height, width, channels, step, tensorType);
            ReadOnlySpan<long> dimensions = stackalloc long[] { length };
            using var buffer = TorchSharpEx.CreateTensorFromUnmanagedMemoryWithManagedAnchor(data, anchor, dimensions, tensorType, device);
            return buffer.as_strided(size, strides);
        }

        private static void CopyTensor(Tensor tensor, IntPtr data, object anchor, int height, int width, int channels, int step, Device device)
        {
            if (height == 0 || width == 0)
                return;

            // Create a temporary tensor backed by the destination memory and copy the source tensor into it
            var (length, size, strides) = GetLayout(height, width, channels, step, tensor.dtype);
            ReadOnlySpan<long> dimensions = stackalloc long[] { length };
            using var buffer = TorchSharpEx.CreateStackTensor(data, anchor, dimensions, tensor.dtype, device);
            using var destination = buffer.Tensor.as_strided(size, strides);
            destination.copy_(tensor);
        }

        /// <summary>
        /// Converts an OpenCV image to a Torch tensor.
        /// </summary>
        /// <param name="image">The image to convert.</param>
        /// <param name="device">The device on which to create the tensor.</param>
        /// <returns>
        /// A tensor with dimensions height, width and channels that shares memory with the image.
        /// The tensor is not contiguous when the rows of the image are padded or the image is
        /// a region of a larger image.
        /// </returns>
        /// <exception cref="NotSupportedException">The depth of the image has no matching tensor type.</exception>
        public static Tensor ToTensor(IplImage image, Device device)
        {
            if (image == null)
                return empty([ 0, 0, 0 ]);

            var tensorType = GetTensorType(image.Depth);
            IntPtr data = image.ImageData;

            if (data == IntPtr.Zero)
                throw new InvalidOperationException($"Got {nameof(IplImage)} without backing data, this isn't expected to be possible.");

            return CreateTensor(data, image, image.Height, image.Width, image.Channels, image.WidthStep, tensorType, device);
        }

        /// <summary>
        /// Converts an OpenCV mat to a Torch tensor.
        /// </summary>
        /// <param name="mat">The matrix to convert.</param>
        /// <param name="device">The device on which to create the tensor.</param>
        /// <returns>
        /// A tensor with dimensions rows, columns and channels that shares memory with the matrix.
        /// The tensor is not contiguous when the matrix is a region of a larger matrix.
        /// </returns>
        /// <exception cref="NotSupportedException">The depth of the matrix has no matching tensor type.</exception>
        public static Tensor ToTensor(Mat mat, Device device)
        {
            if (mat == null)
                return empty([0, 0, 0 ]);

            var tensorType = GetTensorType(mat.Depth);
            IntPtr data = mat.Data;

            if (data == IntPtr.Zero)
                throw new InvalidOperationException($"Got {nameof(Mat)} without backing data, this isn't expected to be possible.");

            return CreateTensor(data, mat, mat.Rows, mat.Cols, mat.Channels, mat.Step, tensorType, device);
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
        /// <param name="tensor">A tensor with dimensions height, width and channels.</param>
        /// <param name="device">The device of the temporary tensor used to copy into the image.</param>
        /// <returns>A new image containing a copy of the tensor.</returns>
        /// <exception cref="NotSupportedException">The type of the tensor has no matching image depth.</exception>
        public static IplImage ToImage(Tensor tensor, Device device)
        {
            var (height, width, channels) = tensor.GetImageDimensions();

            var iplDepth = GetDepth(tensor.dtype).IplDepth;
            var image = new IplImage(new OpenCV.Net.Size(width, height), iplDepth, channels);
            CopyTensor(tensor, image.ImageData, image, height, width, channels, image.WidthStep, device);

            return image;
        }

        /// <summary>
        /// Converts a Torch tensor to an OpenCV mat.
        /// </summary>
        /// <param name="tensor">A tensor with dimensions rows, columns and channels.</param>
        /// <param name="device">The device of the temporary tensor used to copy into the matrix.</param>
        /// <returns>A new matrix containing a copy of the tensor.</returns>
        /// <exception cref="NotSupportedException">The type of the tensor has no matching matrix depth.</exception>
        public static Mat ToMat(Tensor tensor, Device device)
        {
            var (height, width, channels) = tensor.GetImageDimensions();

            var depth = GetDepth(tensor.dtype).Depth;
            var mat = new Mat(new OpenCV.Net.Size(width, height), depth, channels);
            CopyTensor(tensor, mat.Data, mat, height, width, channels, mat.Step, device);

            return mat;
        }
    }
}
