using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Loads a tensor from the specified file.
    /// </summary>
    [Combinator]
    [Description("Loads a tensor from the specified file.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class LoadTensor
    {
        /// <summary>
        /// The path to the file containing the tensor.
        /// </summary>
        [FileNameFilter("Binary files(*.bin)|*.bin|Tensor files(*.pt)|*.pt|All files|*.*")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        [Description("The path to the file containing the tensor.")]
        public string Path { get; set; }

        /// <summary>
        /// Loads a tensor from the specified file.
        /// </summary>
        /// <returns></returns>
        public IObservable<Tensor> Process()
        {
            return Observable.Return(Tensor.Load(Path));
        }
    }
}
