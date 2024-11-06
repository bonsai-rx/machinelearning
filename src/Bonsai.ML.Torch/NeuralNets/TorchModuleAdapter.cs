using System;
using System.Reflection;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.NeuralNets
{
    public class TorchModuleAdapter : ITorchModule
    {
        private readonly nn.Module<Tensor, Tensor> _module = null;

        private readonly jit.ScriptModule<Tensor, Tensor> _scriptModule = null;

        private Func<Tensor, Tensor> forwardFunc;

        public nn.Module Module { get; }

        public TorchModuleAdapter(nn.Module<Tensor, Tensor> module)
        {
            _module = module;
            forwardFunc = _module.forward;
            Module = _module;
        }

        public TorchModuleAdapter(jit.ScriptModule<Tensor, Tensor> scriptModule)
        {
            _scriptModule = scriptModule;
            forwardFunc = _scriptModule.forward;
            Module = _scriptModule;
        }

        public Tensor forward(Tensor input)
        {
            return forwardFunc(input);
        }
    }
}