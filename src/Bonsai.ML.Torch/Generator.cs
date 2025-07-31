using Bonsai;
using static TorchSharp.torch;
using TorchSharp;
using System;
using System.Reactive.Linq;

namespace Bonsai.ML.Torch;

[Combinator]
public class Generator
{
    public Device Device { get; set; }

    public ulong Seed { get; set; } = 0;

    public IObservable<torch.Generator> Process()
    {
        return Observable.Return(new torch.Generator(Seed, Device));
    }
}
