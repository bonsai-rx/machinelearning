using System;
using System.ComponentModel;
using System.Reactive.Linq;
using TorchSharp;
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
        [FileNameFilter("Binary files(*.bin)|*.bin|Tensor files(*.pt)|*.pt|All files|*.*")]
        [Editor("Bonsai.Design.SaveFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        [Description("The path to the file where the tensor will be saved.")]
        public string Path { get; set; } = string.Empty;

        /// <summary>
        /// Indicates whether to use the native torch save method for the tensor.
        /// </summary>
        /// <remarks>
        /// If set to true, the native torch save method will be used. 
        /// If set to false, the tensor will be saved using the TorchSharp method which is specific to .NET formats.
        /// </remarks>
        [Description("Indicates whether to use the native torch save method for the tensor.")]
        public bool UseNativeTorchMethod { get; set; } = false;

        /// <summary>
        /// Saves the input tensor to the specified file.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Do(tensor =>
            {
                if (UseNativeTorchMethod)
                    tensor.save(Path);
                else
                    tensor.Save(Path);
            });
        }
    }
}