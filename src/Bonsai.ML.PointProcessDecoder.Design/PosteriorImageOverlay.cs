using Bonsai;
using Bonsai.Design;
using Bonsai.Expressions;
using System;
using OxyPlot.Series;
using OxyPlot.Axes;
using OxyPlot;
using Bonsai.Vision.Design;
using Bonsai.ML.Torch;
using OpenCV.Net;
using static TorchSharp.torch;
using PointProcessDecoder.Core;
using System.Drawing.Imaging;
using System.Linq;
using TorchSharp;

[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.PosteriorImageOverlay),
    Target = typeof(MashupSource<ImageMashupVisualizer, Bonsai.ML.PointProcessDecoder.Design.PosteriorVisualizer>))]

namespace Bonsai.ML.PointProcessDecoder.Design
{
    /// <summary>
    /// Class that overlays the true posterior distribution on the input image.
    /// </summary>
    public class PosteriorImageOverlay : DialogTypeVisualizer
    {

        private ImageMashupVisualizer imageVisualizer;
        private int[] _stateSpaceMin;
        private int[] _stateSpaceMax;
        private int _height;
        private int _width;
        private PointProcessModel _model;
        private string _modelName;
        private static Func<object, Tensor> _extractPosterior;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            imageVisualizer = (ImageMashupVisualizer)provider.GetService(typeof(MashupVisualizer));
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            if (value is not DecoderDataFrame && value is not ClassifierDataFrame)
            {
                return;
            }

            _modelName ??= value switch
            {
                DecoderDataFrame decoderDataFrame => decoderDataFrame.Name,
                ClassifierDataFrame classifierDataFrame => classifierDataFrame.Name,
                _ => throw new InvalidOperationException("The input value is invalid.")
            };

            _extractPosterior ??= value switch
            {
                DecoderDataFrame _ => input => ((DecoderDataFrame)input).DecoderData.Posterior,
                ClassifierDataFrame _ => input => ((ClassifierDataFrame)input).ClassifierData.DecoderData.Posterior,
                _ => throw new InvalidOperationException("The node is invalid.")
            };

            if (_model is null)
            {
                _model = PointProcessModelManager.GetModel(_modelName);

                _stateSpaceMin = [.. _model.StateSpace.Points
                    .min(dim: 0)
                    .values
                    .to_type(ScalarType.Int32)
                    .data<int>()
                ];

                _stateSpaceMax = [.. _model.StateSpace.Points
                    .max(dim: 0)
                    .values
                    .to_type(ScalarType.Int32)
                    .data<int>()
                ];

                _width = _stateSpaceMax[0] - _stateSpaceMin[0];
                _height = _stateSpaceMax[1] - _stateSpaceMin[1];
            }

            var image = imageVisualizer.VisualizerImage;

            var posterior = _extractPosterior(value)[-1].T.unsqueeze(0);

            var posteriorScaled = torchvision.transforms.functional.resize(posterior, _height, _width);
            posteriorScaled -= posteriorScaled.min();
            posteriorScaled /= posteriorScaled.max();
            posteriorScaled *= 255.0;

            var fullPosterior = zeros([1, image.Height, image.Width], dtype: ScalarType.Byte, device: posterior.device);

            fullPosterior[0, torch.TensorIndex.Slice(_stateSpaceMin[1], _stateSpaceMax[1]), torch.TensorIndex.Slice(_stateSpaceMin[0], _stateSpaceMax[0])] = posteriorScaled.to_type(ScalarType.Byte);

            var posteriorImage = OpenCVHelper.ToImage(fullPosterior.cpu().permute(1, 2, 0), CPU);

            var posteriorOverlay = new IplImage(posteriorImage.Size, posteriorImage.Depth, 3);

            CV.CvtColor(posteriorImage, posteriorOverlay, ColorConversion.Gray2Rgb);
            
            CV.LUT(posteriorOverlay, posteriorOverlay, ColormapExtensions.HotLut);

            CV.AddWeighted(image, 0.8, posteriorOverlay, 0.5, 0, image);
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            imageVisualizer = null;
            _model = null;
            _stateSpaceMin = null;
            _stateSpaceMax = null;
            _modelName = null;
            _extractPosterior = null;
        }
    }

    internal static class ColormapExtensions
    {
        public static Mat HotLut => _hotLut ??= EnsureHotLut();
        private static Mat _hotLut;
        private static Mat EnsureHotLut()
        {
            _hotLut = new Mat(1, 256, Depth.U8, 3);
            for (int i = 0; i < 256; i++)
            {
                double t = i / 255.0;
                double r, g, b;
                if (t < 1.0 / 3.0)
                {
                    r = 3 * t;
                    g = 0;
                    b = 0;
                }
                else if (t < 2.0 / 3.0)
                {
                    r = 1;
                    g = 3 * t - 1;
                    b = 0;
                }
                else
                {
                    r = 1;
                    g = 1;
                    b = 3 * t - 2;
                }
                // Clamp and convert to bytes (BGR order)
                byte R = (byte)Math.Round(r * 255);
                byte G = (byte)Math.Round(g * 255);
                byte B = (byte)Math.Round(b * 255);
                _hotLut[i] = new OpenCV.Net.Scalar(B, G, R);
            }
            return _hotLut;
        }
    }
}