using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Architecture;
    /// <summary>
/// Represents a simple convolutional neural network for the MNIST dataset.
/// </summary>
internal class Mnist : Module<Tensor,Tensor>
{
    private readonly Module<Tensor, Tensor> conv1;
    private readonly Module<Tensor, Tensor> conv2;
    private readonly Module<Tensor, Tensor> fc1;
    private readonly Module<Tensor, Tensor> fc2;
        
    private readonly Module<Tensor, Tensor> pool1;

    private readonly Module<Tensor, Tensor> relu1;
    private readonly Module<Tensor, Tensor> relu2;
    private readonly Module<Tensor, Tensor> relu3;

    private readonly Module<Tensor, Tensor> dropout1;
    private readonly Module<Tensor, Tensor> dropout2;

    private readonly Module<Tensor, Tensor> flatten;
    private readonly Module<Tensor, Tensor> logsm;

    /// <summary>
    /// Constructs a new Mnist model.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="numClasses"></param>
    /// <param name="device"></param>
    public Mnist(string name, int numClasses, Device device = null) : base(name)
    {
        conv1 = Conv2d(1, 32, 3);
        conv2 = Conv2d(32, 64, 3);
        fc1 = Linear(9216, 128);
        fc2 = Linear(128, numClasses);
        
        pool1 = MaxPool2d(kernel_size: [2, 2]);

        relu1 = ReLU();
        relu2 = ReLU();
        relu3 = ReLU();

        dropout1 = Dropout(0.25);
        dropout2 = Dropout(0.5);

        flatten = Flatten();
        logsm = LogSoftmax(1);

        RegisterComponents();

        if (device != null && device.type != DeviceType.CPU)
            this.to(device);
    }

    /// <summary>
    /// Forward pass of the Mnist model.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public override Tensor forward(Tensor input)
    {
        var l11 = conv1.forward(input);
        var l12 = relu1.forward(l11);

        var l21 = conv2.forward(l12);
        var l22 = relu2.forward(l21);
        var l23 = pool1.forward(l22);
        var l24 = dropout1.forward(l23);

        var x = flatten.forward(l24);

        var l31 = fc1.forward(x);
        var l32 = relu3.forward(l31);
        var l33 = dropout2.forward(l32);

        var l41 = fc2.forward(l33);

        return logsm.forward(l41);
    }
}