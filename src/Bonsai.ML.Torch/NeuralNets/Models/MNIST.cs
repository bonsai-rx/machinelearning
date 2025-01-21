using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Models
{
    /// <summary>
    /// Represents a simple convolutional neural network for the MNIST dataset.
    /// </summary>
    public class MNIST : Module<Tensor,Tensor>
    {
        private readonly Module<Tensor, Tensor> conv1 = Conv2d(1, 32, 3);
        private readonly Module<Tensor, Tensor> conv2 = Conv2d(32, 64, 3);
        private readonly Module<Tensor, Tensor> fc1 = Linear(9216, 128);
        private readonly Module<Tensor, Tensor> fc2 = Linear(128, 128);
         
        private readonly Module<Tensor, Tensor> pool1 = MaxPool2d(kernelSize: [2, 2]);

        private readonly Module<Tensor, Tensor> relu1 = ReLU();
        private readonly Module<Tensor, Tensor> relu2 = ReLU();
        private readonly Module<Tensor, Tensor> relu3 = ReLU();

        private readonly Module<Tensor, Tensor> dropout1 = Dropout(0.25);
        private readonly Module<Tensor, Tensor> dropout2 = Dropout(0.5);

        private readonly Module<Tensor, Tensor> flatten = Flatten();
        private readonly Module<Tensor, Tensor> logsm = LogSoftmax(1);

        /// <summary>
        /// Constructs a new MNIST model.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="device"></param>
        public MNIST(string name, Device device = null) : base(name)
        {
            RegisterComponents();

            if (device != null && device.type != DeviceType.CPU)
                this.to(device);
        }

        /// <summary>
        /// Forward pass of the MNIST model.
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
}