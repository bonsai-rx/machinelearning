using Bonsai;
using Bonsai.Design;
using Bonsai.Design.Visualizers;
using System;
using System.Collections.Generic;
using OxyPlot.Series;
using OxyPlot.Axes;
using OxyPlot;
using Bonsai.Vision.Design;
using Bonsai.ML.Design;
using System.Linq;
using TorchSharp;
using OpenCV.Net;

[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.Point2DOverlay),
    Target = typeof(MashupSource<Bonsai.ML.PointProcessDecoder.Design.PosteriorVisualizer, PointVisualizer>))]

[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.Point2DOverlay),
    Target = typeof(MashupSource<Bonsai.ML.PointProcessDecoder.Design.LikelihoodVisualizer, PointVisualizer>))]

namespace Bonsai.ML.PointProcessDecoder.Design
{
    /// <summary>
    /// Class that overlays the true 
    /// </summary>
    public class Point2DOverlay : DialogTypeVisualizer
    {
        internal LineSeries _lineSeries;
        internal ScatterSeries _scatterSeries;
        private int _capacity;
        private int _dataCount;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            dynamic service = provider.GetService(typeof(MashupVisualizer));
            _capacity = service.Capacity;

            _lineSeries = new LineSeries()
            {
                Color = OxyColors.LimeGreen,
                StrokeThickness = 2
            };

            var colorAxis = new LinearColorAxis()
            {
                IsAxisVisible = false,
                Key = "Point2DOverlayColorAxis"
            };

            _scatterSeries = new ScatterSeries()
            {
                MarkerType = MarkerType.Circle,
                MarkerSize = 10,
                MarkerFill = OxyColors.LimeGreen,
                ColorAxisKey = "Point2DOverlayColorAxis"
            };

            
            service.Plot.Model.Series.Add(_scatterSeries);
            service.Plot.Model.Series.Add(_lineSeries);
            service.Plot.Model.Axes.Add(colorAxis);
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            dynamic point = value;
            _dataCount++;
            _lineSeries.Points.Add(new DataPoint(point.X, point.Y));
            _scatterSeries.Points.Clear();
            _scatterSeries.Points.Add(new ScatterPoint(point.X, point.Y, value: 1));

            while (_dataCount > _capacity)
            {
                _lineSeries.Points.RemoveAt(0);
                _dataCount--;
            }
        }

        /// <inheritdoc/>
        public override void Unload()
        {
        }
    }
}