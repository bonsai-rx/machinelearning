using System;
using System.Reactive.Linq;
using System.ComponentModel;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// Represents an operator that slices a 2D multi-dimensional array.
    /// </summary>
    [Combinator]
    [Description("Slices a 2D multi-dimensional array.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Slice
    {

        /// <summary>
        /// Gets or sets the index to begin slicing the rows of the array. A value of null indicates the first row of the array.
        /// </summary>
        [Description("The index to begin slicing the rows of the array. A value of null indicates the first row of the array.")]
        public int? RowStart { get; set; }

        /// <summary>
        /// Gets or sets the index to stop slicing the rows of the array. A value of null indicates the last row of the array.
        /// </summary>
        [Description("The index to stop slicing the rows of the array. A value of null indicates the last row of the array.")]
        public int? RowEnd { get; set; }

        /// <summary>
        /// Gets or sets the index to begin slicing the columns of the array. A value of null indicates the first column of the array.
        /// </summary>
        [Description("The index to begin slicing the columns of the array. A value of null indicates the first column of the array.")]
        public int? ColStart { get; set; }

        /// <summary>
        /// Gets or sets the index to stop slicing the columns of the array. A value of null indicates the last column of the array.
        /// </summary>
        [Description("The index to stop slicing the columns of the array. A value of null indicates the last column of the array.")]
        public int? ColEnd { get; set; }

        /// <summary>
        /// Slices a 2D multi-dimensional array into a new multi-dimensional array by extracting elements between the provided start and end indices of the rows and columns.
        /// </summary>
        public IObservable<double[,]> Process(IObservable<double[,]> source)
        {

            int rowStart = RowStart.HasValue ? RowStart.Value : 0;
            int rowEnd = RowEnd.HasValue ? RowEnd.Value : int.MaxValue;

            int colStart = ColStart.HasValue ? ColStart.Value : 0;
            int colEnd = ColEnd.HasValue ? ColEnd.Value : int.MaxValue;

            if (rowEnd < rowStart)
            {
                throw new InvalidOperationException("Starting row must be less than or equal to ending row.");
            }

            if (colEnd < colStart)
            {
                throw new InvalidOperationException("Starting column must be less than or equal to ending column.");
            }

            return Observable.Select(source, value =>
            {
                var inputRows = value.GetLength(0);
                var inputCols = value.GetLength(1);

                if (rowEnd == int.MaxValue)
                {
                    rowEnd = inputRows;
                }

                if (colEnd == int.MaxValue)
                {
                    colEnd = inputCols;
                }

                int rowCount = rowEnd - rowStart;
                int colCount = colEnd - colStart;

                double[,] slicedArray = new double[rowCount, colCount];

                for (int i = 0; i < rowCount; i++)
                {
                    for (int j = 0; j < colCount; j++)
                    {
                        slicedArray[i, j] = value[rowStart + i, colStart + j];
                    }
                }

                return slicedArray;
            });    
        }
    }
}