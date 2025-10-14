using System;
using System.Linq;
using System.ComponentModel;
using System.Reactive;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;
using System.Threading.Tasks;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Trains a model using backpropagation.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Trains a model using backpropagation.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Backward
{
    /// <summary>
    /// The optimizer to use for training.
    /// </summary>
    [XmlIgnore]
    public optim.Optimizer Optimizer { get; set; } = null;

    /// <summary>
    /// The module to train.
    /// </summary>
    [XmlIgnore]
    public IModule<Tensor, Tensor> Module { get; set; }

    /// <summary>
    /// The criterion (loss function) to use for training.
    /// </summary>
    [XmlIgnore]
    public IModule<Tensor, Tensor, Tensor> Criterion { get; set; } = null;

    /// <summary>
    /// The learning rate scheduler to use for training.
    /// </summary>
    [XmlIgnore]
    public LRScheduler LRScheduler { get; set; } = null;

    /// <summary>
    /// The number of epochs to train the model.
    /// </summary>
    public int Epochs { get; set; } = 1;

    /// <summary>
    /// Trains the model using backpropagation.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Tensor, Tensor>> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(input => Observable.Create<Tuple<Tensor, Tensor>>((observer, cancellationToken) =>
        {
            return Task.Run(() =>
            {
                var module = Module as Module<Tensor, Tensor>;

                Criterion ??= NLLLoss();
                var criterion = Criterion as Module<Tensor, Tensor, Tensor>;

                Optimizer ??= SGD(module.parameters(), 0.01);
                if (!module.training)
                    module.train();

                var (data, target) = input;

                for (int epoch = 0; epoch < Epochs; epoch++)
                {
                    if (cancellationToken.IsCancellationRequested) break;
                    using (_ = NewDisposeScope())
                    {
                        Optimizer.zero_grad();

                        var prediction = module.forward(data);
                        var loss = criterion.forward(prediction, target);

                        loss.backward();

                        Optimizer.step();

                        observer.OnNext(Tuple.Create(prediction, loss));
                    }
                    LRScheduler?.step();
                }
                observer.OnCompleted();
                return Disposable.Empty;
            }, cancellationToken);
        })).Concat();
    }
}