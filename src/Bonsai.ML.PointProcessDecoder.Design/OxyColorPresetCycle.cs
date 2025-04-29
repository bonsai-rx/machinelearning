
using OxyPlot;

namespace Bonsai.ML.PointProcessDecoder.Design
{
    /// <summary>
    /// Enumerates the colors and provides a preset collection of colors to cycle through.
    /// </summary>
    public class OxyColorPresetCycle
    {
        private static readonly OxyColor[] Colors =
        [
            OxyColors.LimeGreen,
            OxyColors.Red,
            OxyColors.Blue,
            OxyColors.Orange,
            OxyColors.Purple,
            OxyColors.Yellow,
            OxyColors.Pink,
            OxyColors.Brown,
            OxyColors.Cyan,
            OxyColors.Magenta,
            OxyColors.Green,
            OxyColors.Gray,
            OxyColors.Black
        ];

        private int _index;

        /// <summary>
        /// Gets the next color in the cycle.
        /// </summary>
        public OxyColor Next()
        {
            var color = Colors[_index];
            _index = (_index + 1) % Colors.Length; 
            return color;
        }
    }
}