using System;
using Python.Runtime;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    public abstract class ObservationParams
    {
        public abstract object[] Params { get; set; }
    }

    public abstract class ObservationParams<T> : ObservationParams where T : ObservationParams<T>
    {
        public abstract override object[] Params { get; set; }
    }
}
