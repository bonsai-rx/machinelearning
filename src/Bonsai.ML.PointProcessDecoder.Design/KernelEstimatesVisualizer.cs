using System;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Linq;
using Bonsai;
using Bonsai.Dag;
using Bonsai.Expressions;
using Bonsai.Design;
using Bonsai.ML.Design;
using PointProcessDecoder.Core;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;

[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.KernelEstimatesVisualizer), 
    Target = typeof(PointProcessModel))]

namespace Bonsai.ML.PointProcessDecoder.Design
{
    public class KernelEstimatesVisualizer : DialogTypeVisualizer
    {
        private PointProcessModel _model = null;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            var expressionBuilderGraph = (ExpressionBuilderGraph)provider.GetService(typeof(ExpressionBuilderGraph));
            var typeVisualizerContext = (ITypeVisualizerContext)provider.GetService(typeof(ITypeVisualizerContext));
            if (expressionBuilderGraph != null && typeVisualizerContext != null)
            {
                _model = ExpressionBuilder.GetWorkflowElement(
                    expressionBuilderGraph.Where(node => node.Value == typeVisualizerContext.Source)
                        .FirstOrDefault().Value) as DensityCluster;
            }

            if (_densityCluster == null)
            {
                throw new InvalidOperationException("Unable to access the density cluster workflow element.");
            }

            if (_densityCluster.Dimensions != 2)
            {
                throw new InvalidOperationException("The density visualizer can only be used with 2 dimensional data.");
            }

            base.Load(provider);

            var showDensityClusterInfoLabel = new ToolStripLabel() 
            {
                Text = "Density Cluster Info: ",
                AutoSize = true
            };

            var showDensityClusterInfoCombobox = new ToolStripComboBox()
            {
                Name = "densityClusterInfoComboBox",
            };

            showDensityClusterInfoCombobox.Items.AddRange([
                "Density Values",
                "Cluster Ids",
                "Density Labels"
            ]);

            showDensityClusterInfoCombobox.SelectedIndexChanged += (sender, e) =>
            {
                var combobox = (ToolStripComboBox)sender;
                var selectedIndex = combobox.SelectedIndex;
                _getDensityInfo = selectedIndex switch
                {
                    0 => _densityCluster.GetCellGridDensities,
                    1 => _densityCluster.GetCellGridClusterIds,
                    2 => _densityCluster.GetCellGridDensityLabels,
                    _ => throw new InvalidOperationException("Invalid density cluster info selection.")
                };
            };
            
            var toolStripItems = new ToolStripItem[] {
                showDensityClusterInfoLabel,
                showDensityClusterInfoCombobox
            };

            _getDensityInfo = _densityCluster.GetCellGridDensities;

            Plot.StatusStrip.Items.AddRange(toolStripItems);
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            var densityInfo = (double[,])_getDensityInfo();
            if (densityInfo == null || densityInfo.Length == 0)
            {
                return;
            }
            base.Show(densityInfo);
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            _densityCluster = null;
            base.Unload();
        }
    }
}