using System;
using System.Reflection;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.NeuralNets
{
    /// <summary>
    /// Represents a torch module adapter that wraps a torch module or script module.
    /// </summary>
    public class TorchModuleAdapter : ITorchModule
    {
        private readonly nn.Module<Tensor, Tensor> _module = null;

        private readonly jit.ScriptModule<Tensor, Tensor> _scriptModule = null;

        private readonly Func<Tensor, Tensor> _forwardFunc;

        /// <summary>
        /// The module.
        /// </summary>
        public nn.Module Module { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="TorchModuleAdapter"/> class.
        /// </summary>
        /// <param name="module"></param>
        public TorchModuleAdapter(nn.Module<Tensor, Tensor> module)
        {
            _module = module;
            _forwardFunc = _module.forward;
            Module = _module;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TorchModuleAdapter"/> class.
        /// </summary>
        /// <param name="scriptModule"></param>
        public TorchModuleAdapter(jit.ScriptModule<Tensor, Tensor> scriptModule)
        {
            _scriptModule = scriptModule;
            _forwardFunc = _scriptModule.forward;
            Module = _scriptModule;
        }

        /// <inheritdoc/>
        public Tensor Forward(Tensor input)
        {
            return _forwardFunc(input);
        }
    }
}