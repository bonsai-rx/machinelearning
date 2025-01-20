using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class LoadModuleFromArchitecture
    {
        public Models.ModelArchitecture ModelArchitecture { get; set; }

        [XmlIgnore]
        public Device Device { get; set; }

        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        public string ModelWeightsPath { get; set; }

        private int numClasses = 10;
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

        public IObservable<ITorchModule> Process()
        {
            var modelArchitecture = ModelArchitecture.ToString().ToLower();
            var device = Device;

            nn.Module<Tensor,Tensor> module = modelArchitecture switch
            {
                "alexnet" => new Models.AlexNet(modelArchitecture, numClasses, device),
                "mobilenet" => new Models.MobileNet(modelArchitecture, numClasses, device),
                "mnist" => new Models.MNIST(modelArchitecture, device),
                _ => throw new ArgumentException($"Model {modelArchitecture} not supported.")
            };

            if (ModelWeightsPath is not null) module.load(ModelWeightsPath);

            var torchModule = new TorchModuleAdapter(module);
            return Observable.Defer(() => {
                return Observable.Return((ITorchModule)torchModule);
            });
        }
    }
}