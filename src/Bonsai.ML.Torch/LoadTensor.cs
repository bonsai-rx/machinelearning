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
        /// Indicates whether to use the native torch load method for the tensor.
        /// </summary>
        /// <remarks>
        /// If set to true, the native torch load method will be used. 
        /// If set to false, the tensor will be loaded using the TorchSharp method which is specific to .NET formats.
        /// </remarks>
        [Description("Indicates whether to use the native torch load method for the tensor.")]
        public bool UseNativeTorchMethod { get; set; } = false;

        /// <summary>
        /// Loads a tensor from the specified file.
        /// </summary>
        /// <returns></returns>
        public IObservable<Tensor> Process()
        {
            switch (UseNativeTorchMethod)
            {
                case true:
                    return Observable.Return(load(Path));
                case false:
                    return Observable.Return(Tensor.Load(Path));
            }
        }
    }
}