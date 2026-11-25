using System.Runtime.InteropServices;
using OpenCV.Net;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Tests;

/// <summary>
/// Tests for the conversions between OpenCV images, OpenCV matrices and tensors.
/// </summary>
[TestClass]
public class OpenCVHelperTests
{
    private const int Height = 4;
    private static string? nativeLibraryError;

    // OpenCV.Net has no native runtime package for Linux, so the tests are reported as
    // inconclusive when the OpenCV libraries are not installed.
    [ClassInitialize]
    public static void LoadNativeLibrary(TestContext context)
    {
        try
        {
            using var image = new IplImage(new OpenCV.Net.Size(1, 1), IplDepth.U8, 1);
        }
        catch (Exception ex) when (ex is DllNotFoundException || ex.InnerException is DllNotFoundException)
        {
            nativeLibraryError = (ex.InnerException ?? ex).Message;
        }
    }

    [TestInitialize]
    public void RequireNativeLibrary()
    {
        if (nativeLibraryError != null)
            Assert.Inconclusive($"The OpenCV native libraries could not be loaded. {nativeLibraryError}");
    }

    private static int ElementSize(IplDepth depth) => depth switch
    {
        IplDepth.U8 or IplDepth.S8 => 1,
        IplDepth.U16 or IplDepth.S16 => 2,
        IplDepth.S32 or IplDepth.F32 => 4,
        IplDepth.F64 => 8,
        _ => throw new ArgumentOutOfRangeException(nameof(depth))
    };

    private static IplDepth ToIplDepth(Depth depth) => depth switch
    {
        Depth.U8 => IplDepth.U8,
        Depth.S8 => IplDepth.S8,
        Depth.U16 => IplDepth.U16,
        Depth.S16 => IplDepth.S16,
        Depth.S32 => IplDepth.S32,
        Depth.F32 => IplDepth.F32,
        Depth.F64 => IplDepth.F64,
        _ => throw new ArgumentOutOfRangeException(nameof(depth))
    };

    private static ScalarType ToScalarType(IplDepth depth) => depth switch
    {
        IplDepth.U8 => ScalarType.Byte,
        IplDepth.S8 => ScalarType.Int8,
        IplDepth.S16 => ScalarType.Int16,
        IplDepth.S32 => ScalarType.Int32,
        IplDepth.F32 => ScalarType.Float32,
        IplDepth.F64 => ScalarType.Float64,
        _ => throw new ArgumentOutOfRangeException(nameof(depth))
    };

    // A value that is distinct for every element of a small image and fits in a signed byte.
    private static double PatternValue(int row, int column, int channel, int width, int channels)
    {
        return (row * width * channels + column * channels + channel) % 113 + 1;
    }

    private static void WriteElement(IntPtr address, IplDepth depth, double value)
    {
        switch (depth)
        {
            case IplDepth.U8: Marshal.WriteByte(address, (byte)value); break;
            case IplDepth.S8: Marshal.WriteByte(address, unchecked((byte)(sbyte)value)); break;
            case IplDepth.U16: Marshal.WriteInt16(address, unchecked((short)(ushort)value)); break;
            case IplDepth.S16: Marshal.WriteInt16(address, (short)value); break;
            case IplDepth.S32: Marshal.WriteInt32(address, (int)value); break;
            case IplDepth.F32: Marshal.WriteInt32(address, BitConverter.SingleToInt32Bits((float)value)); break;
            case IplDepth.F64: Marshal.WriteInt64(address, BitConverter.DoubleToInt64Bits(value)); break;
            default: throw new ArgumentOutOfRangeException(nameof(depth));
        }
    }

    private static double ReadElement(IntPtr address, IplDepth depth)
    {
        return depth switch
        {
            IplDepth.U8 => Marshal.ReadByte(address),
            IplDepth.S8 => unchecked((sbyte)Marshal.ReadByte(address)),
            IplDepth.U16 => unchecked((ushort)Marshal.ReadInt16(address)),
            IplDepth.S16 => Marshal.ReadInt16(address),
            IplDepth.S32 => Marshal.ReadInt32(address),
            IplDepth.F32 => BitConverter.Int32BitsToSingle(Marshal.ReadInt32(address)),
            IplDepth.F64 => BitConverter.Int64BitsToDouble(Marshal.ReadInt64(address)),
            _ => throw new ArgumentOutOfRangeException(nameof(depth))
        };
    }

    // Fills every byte of the buffer, including any row padding, with a marker value
    // and then writes the pattern into the pixels.
    private static void FillBuffer(IntPtr data, int step, int height, int width, int channels, IplDepth depth)
    {
        var elementSize = ElementSize(depth);
        for (int i = 0; i < height * step; i++)
            Marshal.WriteByte(data + i, 0xEE);

        for (int row = 0; row < height; row++)
            for (int column = 0; column < width; column++)
                for (int channel = 0; channel < channels; channel++)
                {
                    var address = data + row * step + (column * channels + channel) * elementSize;
                    WriteElement(address, depth, PatternValue(row, column, channel, width, channels));
                }
    }

    // Reads the pixels of a buffer in row, column, channel order using its row stride.
    private static double[] ReadBuffer(IntPtr data, int step, int height, int width, int channels, IplDepth depth)
    {
        var elementSize = ElementSize(depth);
        var values = new double[height * width * channels];
        var index = 0;
        for (int row = 0; row < height; row++)
            for (int column = 0; column < width; column++)
                for (int channel = 0; channel < channels; channel++)
                    values[index++] = ReadElement(data + row * step + (column * channels + channel) * elementSize, depth);
        return values;
    }

    private static double[] ReadTensor(Tensor tensor)
    {
        using var converted = tensor.to_type(ScalarType.Float64).contiguous();
        return converted.data<double>().ToArray();
    }

    private static string Describe(long[] shape) => $"[{string.Join(", ", shape)}]";

    // Fails when the tensor describes more memory than is available from its first element,
    // so that a test never reads outside of the image buffer.
    private static void AssertTensorFitsBuffer(Tensor tensor, long availableBytes)
    {
        var tensorBytes = tensor.numel() * tensor.dtype.ElementSize();
        Assert.IsTrue(
            tensorBytes <= availableBytes,
            $"The tensor of shape {Describe(tensor.shape)} and type {tensor.dtype} describes {tensorBytes} bytes but only {availableBytes} bytes are available.");
    }

    private static IplImage CreateImage(int width, IplDepth depth, int channels)
    {
        var image = new IplImage(new OpenCV.Net.Size(width, Height), depth, channels);
        FillBuffer(image.ImageData, image.WidthStep, Height, width, channels, depth);
        return image;
    }

    private static Mat CreateMat(int width, Depth depth, int channels)
    {
        var mat = new Mat(Height, width, depth, channels);
        FillBuffer(mat.Data, mat.Step, Height, width, channels, ToIplDepth(depth));
        return mat;
    }

    private static Tensor CreateTensor(int width, int channels, ScalarType type)
    {
        var values = new double[Height * width * channels];
        var index = 0;
        for (int row = 0; row < Height; row++)
            for (int column = 0; column < width; column++)
                for (int channel = 0; channel < channels; channel++)
                    values[index++] = PatternValue(row, column, channel, width, channels);
        return tensor(values, [Height, width, channels]).to_type(type);
    }

    public static IEnumerable<object[]> ImageFormats =>
    [
        // Rows without padding
        [IplDepth.U8, 1, 16],
        [IplDepth.U8, 4, 5],
        [IplDepth.S8, 1, 16],
        [IplDepth.S16, 1, 16],
        [IplDepth.S16, 3, 6],
        [IplDepth.S32, 1, 5],
        [IplDepth.S32, 3, 5],
        [IplDepth.F32, 1, 16],
        [IplDepth.F32, 3, 21],
        [IplDepth.F64, 1, 5],
        [IplDepth.F64, 3, 5],
        // Rows with padding that is a whole number of pixels
        [IplDepth.U8, 1, 22],
        [IplDepth.S8, 1, 21],
        [IplDepth.S16, 1, 21],
        // Rows with padding that is not a whole number of pixels
        [IplDepth.U8, 3, 21],
        [IplDepth.U8, 3, 22],
        [IplDepth.S16, 3, 21],
    ];

    [TestMethod]
    [DynamicData(nameof(ImageFormats))]
    public void ToTensor_Image_ShapeMatchesImage(IplDepth depth, int channels, int width)
    {
        using var image = CreateImage(width, depth, channels);
        using var result = OpenCVHelper.ToTensor(image, CPU);

        Assert.AreEqual(
            Describe([Height, width, channels]),
            Describe(result.shape),
            $"WidthStep is {image.WidthStep}.");
        Assert.AreEqual(ToScalarType(depth), result.dtype);
    }

    [TestMethod]
    [DynamicData(nameof(ImageFormats))]
    public void ToTensor_Image_ContentMatchesImage(IplDepth depth, int channels, int width)
    {
        using var image = CreateImage(width, depth, channels);
        using var result = OpenCVHelper.ToTensor(image, CPU);

        AssertTensorFitsBuffer(result, Height * image.WidthStep);
        var expected = ReadBuffer(image.ImageData, image.WidthStep, Height, width, channels, depth);
        CollectionAssert.AreEqual(expected, ReadTensor(result), $"WidthStep is {image.WidthStep}.");
    }

    [TestMethod]
    [DataRow(IplDepth.U8, 1)]
    [DataRow(IplDepth.U8, 3)]
    [DataRow(IplDepth.F32, 1)]
    public void ToTensor_CroppedImage_MatchesCroppedRegion(IplDepth depth, int channels)
    {
        using var parent = CreateImage(16, depth, channels);
        var region = new Rect(4, 1, 8, 2);
        using var cropped = parent.GetSubRect(region);
        using var result = OpenCVHelper.ToTensor(cropped, CPU);

        Assert.AreEqual(
            Describe([region.Height, region.Width, channels]),
            Describe(result.shape),
            $"WidthStep of the cropped image is {cropped.WidthStep}.");

        var parentEnd = parent.ImageData.ToInt64() + Height * parent.WidthStep;
        AssertTensorFitsBuffer(result, parentEnd - cropped.ImageData.ToInt64());
        var expected = ReadBuffer(cropped.ImageData, cropped.WidthStep, region.Height, region.Width, channels, depth);
        CollectionAssert.AreEqual(expected, ReadTensor(result));
    }

    [TestMethod]
    public void ToTensor_CroppedImage_DoesNotDescribeMemoryPastParentBuffer()
    {
        using var parent = CreateImage(16, IplDepth.U8, 1);
        using var cropped = parent.GetSubRect(new Rect(4, 1, 8, 3));
        using var result = OpenCVHelper.ToTensor(cropped, CPU);

        var parentEnd = parent.ImageData.ToInt64() + Height * parent.WidthStep;
        AssertTensorFitsBuffer(result, parentEnd - cropped.ImageData.ToInt64());
    }

    [TestMethod]
    [DataRow(Depth.U8, 1, 16)]
    [DataRow(Depth.U8, 1, 22)]
    [DataRow(Depth.U8, 3, 21)]
    [DataRow(Depth.S8, 1, 21)]
    [DataRow(Depth.S16, 3, 21)]
    [DataRow(Depth.S32, 1, 5)]
    [DataRow(Depth.F32, 3, 21)]
    [DataRow(Depth.F64, 1, 5)]
    public void ToTensor_Mat_MatchesMat(Depth depth, int channels, int width)
    {
        using var mat = CreateMat(width, depth, channels);
        using var result = OpenCVHelper.ToTensor(mat, CPU);

        Assert.AreEqual(Describe([Height, width, channels]), Describe(result.shape), $"Step is {mat.Step}.");
        Assert.AreEqual(ToScalarType(ToIplDepth(depth)), result.dtype);
        AssertTensorFitsBuffer(result, Height * mat.Step);
        var expected = ReadBuffer(mat.Data, mat.Step, Height, width, channels, ToIplDepth(depth));
        CollectionAssert.AreEqual(expected, ReadTensor(result));
    }

    [TestMethod]
    [DataRow(Depth.U8, 1)]
    [DataRow(Depth.U8, 3)]
    [DataRow(Depth.F32, 1)]
    public void ToTensor_SubMat_MatchesSubRegion(Depth depth, int channels)
    {
        using var parent = CreateMat(16, depth, channels);
        var region = new Rect(4, 1, 8, 2);
        using var subMat = parent.GetSubRect(region);
        using var result = OpenCVHelper.ToTensor(subMat, CPU);

        Assert.AreEqual(
            Describe([region.Height, region.Width, channels]),
            Describe(result.shape),
            $"Step of the sub-matrix is {subMat.Step}.");

        var parentEnd = parent.Data.ToInt64() + Height * parent.Step;
        AssertTensorFitsBuffer(result, parentEnd - subMat.Data.ToInt64());
        var expected = ReadBuffer(subMat.Data, subMat.Step, region.Height, region.Width, channels, ToIplDepth(depth));
        CollectionAssert.AreEqual(expected, ReadTensor(result));
    }

    [TestMethod]
    [DataRow(IplDepth.U8, 1)]
    [DataRow(IplDepth.U8, 3)]
    [DataRow(IplDepth.S16, 1)]
    public void ToTensor_SingleRowOfImage_MatchesRow(IplDepth depth, int channels)
    {
        using var parent = CreateImage(21, depth, channels);
        var region = new Rect(3, 2, 10, 1);
        using var cropped = parent.GetSubRect(region);
        using var result = OpenCVHelper.ToTensor(cropped, CPU);

        Assert.AreEqual(Describe([1, region.Width, channels]), Describe(result.shape));
        var expected = ReadBuffer(cropped.ImageData, parent.WidthStep, 1, region.Width, channels, depth);
        CollectionAssert.AreEqual(expected, ReadTensor(result));
    }

    [TestMethod]
    [DataRow(Depth.U8, 1)]
    [DataRow(Depth.U8, 3)]
    [DataRow(Depth.S16, 1)]
    public void ToTensor_SingleRowOfMat_MatchesRow(Depth depth, int channels)
    {
        using var parent = CreateMat(21, depth, channels);
        var region = new Rect(3, 2, 10, 1);
        using var subMat = parent.GetSubRect(region);
        using var result = OpenCVHelper.ToTensor(subMat, CPU);

        Assert.AreEqual(Describe([1, region.Width, channels]), Describe(result.shape));
        var expected = ReadBuffer(subMat.Data, parent.Step, 1, region.Width, channels, ToIplDepth(depth));
        CollectionAssert.AreEqual(expected, ReadTensor(result));
    }

    [TestMethod]
    [DataRow(16, true)]
    [DataRow(22, false)]
    public void ToTensor_Image_IsContiguousOnlyWithoutPadding(int width, bool contiguous)
    {
        using var image = CreateImage(width, IplDepth.U8, 1);
        using var result = OpenCVHelper.ToTensor(image, CPU);

        Assert.AreEqual(contiguous, result.is_contiguous());
    }

    // Operators such as ConvertToArray read tensors through the data accessor.
    [TestMethod]
    [DataRow(22, 1)]
    [DataRow(21, 3)]
    public void ToTensor_PaddedImage_DataAccessorMatchesImage(int width, int channels)
    {
        using var image = CreateImage(width, IplDepth.U8, channels);
        using var result = OpenCVHelper.ToTensor(image, CPU);

        var expected = ReadBuffer(image.ImageData, image.WidthStep, Height, width, channels, IplDepth.U8);
        var actual = result.data<byte>().ToArray().Select(value => (double)value).ToArray();
        CollectionAssert.AreEqual(expected, actual);
    }

    [TestMethod]
    [DataRow(16)]
    [DataRow(22)]
    public void ToTensor_Image_SharesMemoryWithImage(int width)
    {
        using var image = CreateImage(width, IplDepth.U8, 1);
        using var result = OpenCVHelper.ToTensor(image, CPU);

        var row = Height - 1;
        var column = width - 1;
        Marshal.WriteByte(image.ImageData + row * image.WidthStep + column, 200);
        Assert.AreEqual(200, result[row, column, 0].ToByte());

        using var value = tensor((byte)100);
        result[0, 0, 0] = value;
        Assert.AreEqual(100, Marshal.ReadByte(image.ImageData));
    }

    [TestMethod]
    public void ToTensor_Mat_SharesMemoryWithMat()
    {
        using var parent = CreateMat(16, Depth.U8, 1);
        using var subMat = parent.GetSubRect(new Rect(4, 1, 8, 2));
        using var result = OpenCVHelper.ToTensor(subMat, CPU);

        Marshal.WriteByte(subMat.Data + parent.Step + 7, 200);
        Assert.AreEqual(200, result[1, 7, 0].ToByte());
    }

    [TestMethod]
    public void ToTensor_Image_RemainsValidAfterImageIsCollected()
    {
        const int width = 22;
        double[] expected;
        Tensor result;
        {
            var image = CreateImage(width, IplDepth.U8, 1);
            expected = ReadBuffer(image.ImageData, image.WidthStep, Height, width, 1, IplDepth.U8);
            result = OpenCVHelper.ToTensor(image, CPU);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        using (result)
            CollectionAssert.AreEqual(expected, ReadTensor(result));
    }

    [TestMethod]
    public void ToTensor_ImageWithUnsupportedDepth_Throws()
    {
        using var image = CreateImage(8, IplDepth.U16, 1);
        Assert.ThrowsException<NotSupportedException>(() => OpenCVHelper.ToTensor(image, CPU));
    }

    [TestMethod]
    public void ToTensor_MatWithUnsupportedDepth_Throws()
    {
        using var mat = CreateMat(8, Depth.U16, 1);
        Assert.ThrowsException<NotSupportedException>(() => OpenCVHelper.ToTensor(mat, CPU));
    }

    [TestMethod]
    public void ToImage_TensorWithUnsupportedType_Throws()
    {
        using var source = zeros([Height, 8, 1], dtype: ScalarType.Int64);
        Assert.ThrowsException<NotSupportedException>(() => OpenCVHelper.ToImage(source, CPU));
    }

    [TestMethod]
    public void ToMat_TensorWithUnsupportedType_Throws()
    {
        using var source = zeros([Height, 8, 1], dtype: ScalarType.Int64);
        Assert.ThrowsException<NotSupportedException>(() => OpenCVHelper.ToMat(source, CPU));
    }

    [TestMethod]
    public void ToImage_NonContiguousTensor_MatchesTensor()
    {
        const int width = 21;
        using var original = CreateTensor(width, 3, ScalarType.Byte);
        using var source = original.flip(1);
        using var image = OpenCVHelper.ToImage(source, CPU);

        var actual = ReadBuffer(image.ImageData, image.WidthStep, Height, width, 3, IplDepth.U8);
        CollectionAssert.AreEqual(ReadTensor(source), actual);
    }

    [TestMethod]
    [DynamicData(nameof(ImageFormats))]
    public void ToImage_MatchesTensor(IplDepth depth, int channels, int width)
    {
        using var source = CreateTensor(width, channels, ToScalarType(depth));
        using var image = OpenCVHelper.ToImage(source, CPU);

        Assert.AreEqual(width, image.Width);
        Assert.AreEqual(Height, image.Height);
        Assert.AreEqual(channels, image.Channels);
        Assert.AreEqual(depth, image.Depth);
        var actual = ReadBuffer(image.ImageData, image.WidthStep, Height, width, channels, depth);
        CollectionAssert.AreEqual(ReadTensor(source), actual, $"WidthStep is {image.WidthStep}.");
    }

    [TestMethod]
    [DynamicData(nameof(ImageFormats))]
    public void ToMat_MatchesTensor(IplDepth depth, int channels, int width)
    {
        using var source = CreateTensor(width, channels, ToScalarType(depth));
        using var mat = OpenCVHelper.ToMat(source, CPU);

        Assert.AreEqual(width, mat.Cols);
        Assert.AreEqual(Height, mat.Rows);
        Assert.AreEqual(channels, mat.Channels);
        Assert.AreEqual(depth, ToIplDepth(mat.Depth));
        var actual = ReadBuffer(mat.Data, mat.Step, Height, width, channels, depth);
        CollectionAssert.AreEqual(ReadTensor(source), actual, $"Step is {mat.Step}.");
    }

    [TestMethod]
    [DynamicData(nameof(ImageFormats))]
    public void ToTensor_ToImage_RoundTripPreservesImage(IplDepth depth, int channels, int width)
    {
        using var image = CreateImage(width, depth, channels);
        using var intermediate = OpenCVHelper.ToTensor(image, CPU);
        AssertTensorFitsBuffer(intermediate, Height * image.WidthStep);
        using var result = OpenCVHelper.ToImage(intermediate, CPU);

        Assert.AreEqual(width, result.Width);
        Assert.AreEqual(Height, result.Height);
        Assert.AreEqual(channels, result.Channels);
        var expected = ReadBuffer(image.ImageData, image.WidthStep, Height, width, channels, depth);
        var actual = ReadBuffer(result.ImageData, result.WidthStep, Height, width, channels, depth);
        CollectionAssert.AreEqual(expected, actual);
    }
}
