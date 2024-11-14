using System;
using System.Collections.Generic;
using System.Windows.Forms;
using Bonsai;
using Bonsai.Design;

[assembly: TypeVisualizer(typeof(Bonsai.ML.Design.UnidimensionalArrayTimeSeriesVisualizer),
    Target = typeof(double[]))]

namespace Bonsai.ML.Design
{
    /// <summary>
    /// Provides a type visualizer to display unidimensional array data as a heatmap time series.
    /// </summary>
    public class UnidimensionalArrayTimeSeriesVisualizer : DialogTypeVisualizer
    {
        /// <summary>
        /// Gets or sets the selected index of the color palette to use.
        /// </summary>
        public int PaletteSelectedIndex { get; set; }

        /// <summary>
        /// Gets or sets the selected index of the render method to use.
        /// </summary>
        public int RenderMethodSelectedIndex { get; set; }

        private HeatMapSeriesOxyPlotBase Plot;

        private int _capacity = 100;

        public int Capacity 
        { 
            get => _capacity;
            set 
            {
                _capacity = value;
                // if (_capacity < dataList.Count)
                // {
                //     dataList.RemoveRange(0, dataList.Count - _capacity);
                // }
            } 
        }

        private List<double[]> dataList = new();

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            Plot = new HeatMapSeriesOxyPlotBase(PaletteSelectedIndex, RenderMethodSelectedIndex)
            {
                Dock = DockStyle.Fill,
            };

            Plot.PaletteComboBoxValueChanged += PaletteIndexChanged;
            Plot.RenderMethodComboBoxValueChanged += RenderMethodIndexChanged;
            
            var capacityLabel = new ToolStripLabel
            {
                Text = "Capacity:",
                AutoSize = true
            };
            var capacityValue = new ToolStripLabel
            {
                Text = Capacity.ToString(),
                AutoSize = true
            };

            Plot.StatusStrip.Items.AddRange([
                capacityLabel,
                capacityValue
            ]);

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            if (visualizerService != null)
            {
                visualizerService.AddControl(Plot);
            }
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            var array = (double[])value;

            if (dataList.Count < Capacity)
            {
                dataList.Add(array);
            }
            else
            {
                while (dataList.Count >= Capacity)
                {
                    dataList.RemoveAt(0);
                }
                dataList.Add(array);
            }

            var shape = (dataList.Count, array.Length);
            var mdarray = new double[shape.Item1, shape.Item2];
            for (int i = 0; i < shape.Item1; i++)
            {
                for (int j = 0; j < shape.Item2; j++)
                {
                    mdarray[i, j] = dataList[i][j];
                }
            }

            Plot.UpdateHeatMapSeries(
                -0.5,
                shape.Item1 - 0.5,
                -0.5,
                shape.Item2 - 0.5,
                mdarray
            );

            Plot.UpdatePlot();
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            if (!Plot.IsDisposed)
            {
                Plot.Dispose();
            }
        }

        private void PaletteIndexChanged(object sender, EventArgs e)
        {
            PaletteSelectedIndex = Plot.PaletteComboBox.SelectedIndex;
        }
        
        private void RenderMethodIndexChanged(object sender, EventArgs e)
        {
            RenderMethodSelectedIndex = Plot.RenderMethodComboBox.SelectedIndex;
        }
    }
}
