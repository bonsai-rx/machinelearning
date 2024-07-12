using Bonsai.Design;
using Bonsai.Vision.Design;
using Bonsai;
using Bonsai.ML.Visualizers.LinearDynamicalSystems;
using Bonsai.ML.LinearDynamicalSystems.Kinematics;
using System;
using System.Collections.Generic;
using OpenCV.Net;
using MathNet.Numerics.LinearAlgebra;
using OxyPlot;

[assembly: TypeVisualizer(typeof(ForecastImageOverlay), Target = typeof(MashupSource<ImageMashupVisualizer, ForecastVisualizer>))]

namespace Bonsai.ML.Visualizers.LinearDynamicalSystems
{
    /// <summary>
    /// Provides a mashup visualizer to display the forecast of a Kalman Filter kinematics model overtime of an ImageMashupVisualizer.
    /// </summary>
    public class ForecastImageOverlay : DialogTypeVisualizer
    {
        private ImageMashupVisualizer visualizer;
        private IplImage overlay;

        /// <inheritdoc/>
        public override void Show(object value)
        {

            var image = visualizer.VisualizerImage;
            Size size = new Size(image.Width, image.Height);
            IplDepth depth = image.Depth;
            int channels = image.Channels;

            overlay = new IplImage(size, depth, channels);
            var alpha = 0.1;

            Forecast forecast = (Forecast)value;
            List<ForecastResult> forecastResults = forecast.ForecastResults;

            for (int i = 0; i < forecastResults.Count; i++)
            {
                var forecastResult = forecastResults[i];
                var kinematicState = forecastResult.KinematicState;

                double xMean = kinematicState.Position.X.Mean;
                double yMean = kinematicState.Position.Y.Mean;

                Point center = new Point((int)Math.Round(xMean), (int)Math.Round(yMean));

                double xVar = kinematicState.Position.X.Variance;
                double yVar = kinematicState.Position.Y.Variance;
                double xyCov = kinematicState.Position.Covariance;

                EllipseParameters ellipseParameters = EllipseHelper.GetEllipseParameters(xVar, yVar, xyCov);

                Size axes = new Size
                {
                    Width = (int)(2 * ellipseParameters.MajorAxis),
                    Height = (int)(2 * ellipseParameters.MinorAxis)
                };

                OxyColor color = OxyColors.Yellow;

                CV.Ellipse(overlay, center, axes, ellipseParameters.Angle, 0, 360, new Scalar(color.B, color.G, color.R, color.A), -1);
            }

            CV.AddWeighted(image, 1 - alpha, overlay, alpha, 1, image);
            overlay.SetZero();
        }
        
        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            visualizer = (ImageMashupVisualizer)provider.GetService(typeof(MashupVisualizer));
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            overlay.Dispose();
        }
    }
}