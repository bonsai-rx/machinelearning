using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Saves the input tensor to the specified file.
    /// </summary>
    [Combinator]
    [Description("Saves the input tensor to the specified file.")]
    [WorkflowElementCategory(ElementCategory.Sink)]
    public class SaveTensor
    {
        /// <summary>
        /// The path to the file where the tensor will be saved.
        /// </summary>
        [FileNameFilter("Binary files(*.bin)|*.bin|All files|*.*")]
        [Editor("Bonsai.Design.SaveFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        [Description("The path to the file where the tensor will be saved.")]
        public string Path { get; set; } = string.Empty;

        /// <summary>
        /// Saves the input tensor to the specified file.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Do(tensor => tensor.save(Path));
        }
    }
}