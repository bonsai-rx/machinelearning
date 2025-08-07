using System.ComponentModel;
using System;
using Bonsai;
using Bonsai.Expressions;

namespace Bonsai.ML.PCA
{
    class PCADescriptionProvider : TypeDescriptionProvider
    {
        private readonly TypeDescriptionProvider _baseProvider;

        public PCADescriptionProvider() : this(TypeDescriptor.GetProvider(typeof(object))) { }

        public PCADescriptionProvider(TypeDescriptionProvider baseProvider)
            : base(baseProvider)
        {
            _baseProvider = baseProvider;
        }

        public override ICustomTypeDescriptor GetTypeDescriptor(Type objectType, object instance)
        {
            var defaultDescriptor = base.GetTypeDescriptor(objectType, instance);
            return new PCADescriptor(defaultDescriptor, instance);
        }
    }
}