using Bonsai;
using Bonsai.Design;
using System;
using OxyPlot.Series;
using OxyPlot.Axes;
using OxyPlot;
using Bonsai.Vision.Design;
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
        private LineSeries _lineSeries;
        private ScatterSeries _scatterSeries;
        private int _dataCount;
        private IDecoderVisualizer decoderVisualizer;
        private OxyColor _color = OxyColors.LimeGreen;
        private Func<object, Point2d> _convertInputData = null;

        /// <summary>
        /// Gets or sets the color of the overlay.
        /// </summary>
        public OxyColor Color 
        { 
            get => _color; 
            set
            {
                if (_lineSeries != null && _scatterSeries != null)
                {
                    _lineSeries.Color = value;
                    _scatterSeries.MarkerFill = value;
                    _color = value;
                }
            }
        }

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            decoderVisualizer = provider.GetService(typeof(MashupVisualizer)) as IDecoderVisualizer;

            _lineSeries = new LineSeries()
            {
                Color = _color,
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
                MarkerFill = _color,
                ColorAxisKey = "Point2DOverlayColorAxis"
            };

            decoderVisualizer.Plot.Model.Series.Add(_scatterSeries);
            decoderVisualizer.Plot.Model.Series.Add(_lineSeries);
            decoderVisualizer.Plot.Model.Axes.Add(colorAxis);
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            if (value == null)
            {
                return;
            }

            if (_convertInputData == null)
            {
                if (value is Point2d)
                {
                    _convertInputData = (data) => (Point2d)data;
                }
                else if (value is Point2f)
                {
                    _convertInputData = (data) => 
                    {
                        var datapoint = (Point2f)data;
                        return new Point2d(datapoint.X, datapoint.Y);
                    };
                }
                else if (value is Point)
                {
                    _convertInputData = (data) => 
                    {
                        var datapoint = (Point)data;
                        return new Point2d(datapoint.X, datapoint.Y);
                    };
                }
            }

            var dataPoint = _convertInputData(value);
            _dataCount++;
            _lineSeries.Points.Add(new DataPoint(dataPoint.X, dataPoint.Y));
            _scatterSeries.Points.Clear();
            _scatterSeries.Points.Add(new ScatterPoint(dataPoint.X, dataPoint.Y, value: 1));

            while (_dataCount > decoderVisualizer.Capacity)
            {
                _lineSeries.Points.RemoveAt(0);
                _dataCount--;
            }
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            _convertInputData = null;
            _dataCount = 0;
            _lineSeries.Points.Clear();
            _scatterSeries.Points.Clear();
        }
    }
}