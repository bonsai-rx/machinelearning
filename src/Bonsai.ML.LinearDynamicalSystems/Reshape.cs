using System;
using System.Reactive.Linq;
using System.ComponentModel;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// Represents an operator that reshapes the dimensions of a 2D multi-dimensional array.
    /// </summary>
    [Combinator]
    [Description("Reshapes the dimensions of a 2D multi-dimensional array.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Reshape
    {
        /// <summary>
        /// Gets or sets the number of rows in the reshaped array.
        /// </summary>
        [Description("The number of rows in the reshaped array.")]
        public int Rows { get; set; }

        /// <summary>
        /// Gets or sets the number of columns in the reshaped array.
        /// </summary>
        [Description("The number of columns in the reshaped array.")]
        public int Cols { get; set; }

        /// <summary>
        /// Reshapes a 2D multi-dimensional array into a new multi-dimensional array with the provided number of rows and columns.
        /// </summary>
        public IObservable<double[,]> Process(IObservable<double[,]> source)
        {

            var rows = Rows;
            var cols = Cols;

            return Observable.Select(source, value =>
            {
                var inputRows = value.GetLength(0);
                var inputCols = value.GetLength(1);
                int totalElements = inputRows * inputCols;

                if (totalElements != rows * cols)
                {
                    throw new InvalidOperationException($"Multi-dimensional array of shape {rows}x{cols} cannot be made from the input array with a total of {totalElements} elements.");
                }

                double[,] reshapedArray = new double[rows, cols];

                for (int i = 0; i < totalElements; i++)
                {
                    int originalRow = i / inputCols;
                    int originalCol = i % inputCols;

                    int newRow = i / cols;
                    int newCol = i % cols;

                    reshapedArray[newRow, newCol] = value[originalRow, originalCol];
                }

                return reshapedArray;
            });    
        }
    }
}