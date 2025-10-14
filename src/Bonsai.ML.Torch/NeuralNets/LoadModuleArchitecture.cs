using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets
{
    /// <summary>
    /// Loads a neural network module from a specified architecture.
    /// </summary>
    [Combinator]
    [ResetCombinator]
    [Description("Loads a neural network module from a specified architecture.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class LoadModuleArchitecture
    {
        /// <summary>
        /// The model architecture to load.
        /// </summary>
        [Description("The model architecture to load.")]
        public Architecture.ModelArchitecture ModelArchitecture { get; set; }

        /// <summary>
        /// The device on which to load the model.
        /// </summary>
        [Description("The device on which to load the model.")]
        [XmlIgnore]
        public Device Device { get; set; }

        /// <summary>
        /// The optional path to the model weights file.
        /// </summary>
        [Description("The optional path to the model weights file.")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        public string ModelWeightsPath { get; set; }

        private int numClasses = 10;
        /// <summary>
        /// The number of classes in the dataset.
        /// </summary>
        [Description("The number of classes in the dataset.")]
        public int NumClasses
        {
            get => numClasses;
            set
            {
                if (value <= 0) 
                {
                    numClasses = 10;
                }
                else
                {
                    numClasses = value;
                }
            }
        }

        /// <summary>
        /// Loads the neural network module from the specified architecture.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public IObservable<IModule<Tensor, Tensor>> Process()
        {
            var device = Device;
            Module<Tensor,Tensor> module = ModelArchitecture switch
            {
                Architecture.ModelArchitecture.AlexNet => new Architecture.AlexNet("alexnet", numClasses, device),
                Architecture.ModelArchitecture.MobileNet => new Architecture.MobileNet("mobilenet", numClasses, device),
                Architecture.ModelArchitecture.Mnist => new Architecture.Mnist("mnist", numClasses, device),
                _ => throw new ArgumentException($"Model {ModelArchitecture} not supported.")
            };

            if (ModelWeightsPath is not null) module.load(ModelWeightsPath);

            return Observable.Return(module);
        }
    }
}