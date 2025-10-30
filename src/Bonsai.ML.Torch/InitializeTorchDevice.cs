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
    [WorkflowElementCategory(ElementCategory.Source)]
    public class InitializeTorchDevice
    {
        /// <summary>
        /// The device type to initialize.
        /// </summary>
        [Description("The device type to initialize.")]
        public DeviceType DeviceType { get; set; }

        /// <summary>
        /// The index of the device to initialize.
        /// </summary>
        [Description("The index of the device to initialize.")]
        public int DeviceIndex { get; set; } = -1;

        /// <summary>
        /// Initializes the Torch device with the specified device type.
        /// </summary>
        /// <returns></returns>
        public IObservable<Device> Process()
        {
            return Observable.Defer(() =>
            {
                InitializeDeviceType(DeviceType);
                return Observable.Return(new Device(DeviceType, DeviceIndex));
            });
        }

        /// <summary>
        /// Initializes the Torch device when the input sequence produces an element.
        /// </summary>
        /// <returns></returns>
        public IObservable<Device> Process<T>(IObservable<T> source)
        {
            return source.Select((_) =>
            {
                InitializeDeviceType(DeviceType);
                return new Device(DeviceType, DeviceIndex);
            });
        }
    }
}