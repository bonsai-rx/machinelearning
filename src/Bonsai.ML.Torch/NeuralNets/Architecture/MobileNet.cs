using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Architecture;

/// <summary>
/// MobileNet model.
/// </summary>
internal class MobileNet : Module<Tensor, Tensor>
{
    private readonly long[] planes = [ 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024 ];
    private readonly long[] strides = [ 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1 ];

    private readonly Module<Tensor, Tensor> layers;

    /// <summary>
    /// Constructs a new MobileNet model.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="numClasses"></param>
    /// <param name="device"></param>
    /// <exception cref="ArgumentException"></exception>
    public MobileNet(string name, int numClasses, Device device = null) : base(name)
    {
        if (planes.Length != strides.Length) throw new ArgumentException("'planes' and 'strides' must have the same length.");

        var modules = new List<(string, Module<Tensor, Tensor>)>
        {
            ($"conv2d-first", Conv2d(3, 32, kernel_size: 3, stride: 1, padding: 1, bias: false)),
            ($"bnrm2d-first", BatchNorm2d(32)),
            ($"relu-first", ReLU())
        };
        MakeLayers(modules, 32);
        modules.Add(("avgpool", AvgPool2d([2, 2])));
        modules.Add(("flatten", Flatten()));
        modules.Add(($"linear", Linear(planes[planes.Length-1], numClasses)));

        layers = Sequential(modules);

        RegisterComponents();

        if (device != null && device.type != DeviceType.CPU)
            this.to(device);
    }

    private void MakeLayers(List<(string, Module<Tensor, Tensor>)> modules, long in_planes)
    {

        for (var i = 0; i < strides.Length; i++) {
            var out_planes = planes[i];
            var stride = strides[i];

            modules.Add(($"conv2d-{i}a", Conv2d(in_planes, in_planes, kernel_size: 3, stride: stride, padding: 1, groups: in_planes, bias: false)));
            modules.Add(($"bnrm2d-{i}a", BatchNorm2d(in_planes)));
            modules.Add(($"relu-{i}a", ReLU()));
            modules.Add(($"conv2d-{i}b", Conv2d(in_planes, out_planes, kernel_size: 1L, stride: 1L, padding: 0L, bias: false)));
            modules.Add(($"bnrm2d-{i}b", BatchNorm2d(out_planes)));
            modules.Add(($"relu-{i}b", ReLU()));

            in_planes = out_planes;
        }
    }

    /// <summary>
    /// Forward pass of the MobileNet model.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public override Tensor forward(Tensor input)
    {
        return layers.forward(input);
    }
}