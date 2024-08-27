using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using TorchSharp;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Initializes the Torch device with the specified device type.
    /// </summary>
    [Combinator]
    [Description("Initializes the Torch device with the specified device type.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class InitializeTorchDevice
    {
        /// <summary>
        /// The device type to initialize.
        /// </summary>
        public DeviceType DeviceType { get; set; }

        /// <summary>
        /// Initializes the Torch device with the specified device type.
        /// </summary>
        /// <returns></returns>
        public IObservable<Device> Process()
        {
            return Observable.Defer(() =>
            {
                InitializeDeviceType(DeviceType);
                return Observable.Return(new Device(DeviceType));
            });
        }
    }
}