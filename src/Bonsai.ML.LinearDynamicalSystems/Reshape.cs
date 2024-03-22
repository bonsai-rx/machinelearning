using System;
using System.Reactive.Linq;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// A class that converts a python object, representing a multivariate PDF, into a multidimensional array
    /// /// </summary>
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Reshape
    {
        public int Rows {get;set;}
        public int Cols {get;set;}

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
                    throw new InvalidOperationException($"Multidimensional array of shape {rows}x{cols} cannot be made from input array with a total of {totalElements} elements.");
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