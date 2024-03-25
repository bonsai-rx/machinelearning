using System;
using System.Reactive.Linq;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// A class that converts a python object, representing a multivariate PDF, into a multidimensional array
    /// /// </summary>
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Slice
    {
        public int? RowStart {get;set;} = null;
        public int? RowEnd {get;set;} = null;
        public int? ColStart {get;set;} = null;
        public int? ColEnd {get;set;} = null;

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

                // Calculate the size of the new sliced array
                int rowCount = rowEnd - rowStart;
                int colCount = colEnd - colStart;

                // Initialize the new array with the calculated size
                double[,] slicedArray = new double[rowCount, colCount];

                // Copy the data from the original array to the new array
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