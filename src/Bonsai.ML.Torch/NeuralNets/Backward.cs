using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Backward
    {
        public Optimizer Optimizer { get; set; }

        [XmlIgnore]
        public ITorchModule Model { get; set; }

        public Loss Loss { get; set; }

        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor>> source)
        {
            optim.Optimizer optimizer = null;
            switch (Optimizer)
            {
                case Optimizer.Adam:
                    optimizer = Adam(Model.Module.parameters());
                    break;
            }

            Module<Tensor, Tensor, Tensor> loss = null;
            switch (Loss)
            {
                case Loss.NLLLoss:
                    loss = NLLLoss();
                    break;
            }

            var scheduler = lr_scheduler.StepLR(optimizer, 1, 0.7);
            Model.Module.train();

            return source.Select((input) => {
                var (data, target) = input;
                using (_ = NewDisposeScope())
                {
                    optimizer.zero_grad();
                    
                    var prediction = Model.forward(data);
                    var output = loss.forward(prediction, target);

                    output.backward();

                    optimizer.step();
                    return output.MoveToOuterDisposeScope();
                }
            });
        }
    }
}