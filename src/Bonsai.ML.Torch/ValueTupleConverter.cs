using System;
using System.ComponentModel;

namespace Bonsai.ML.Torch;

/// <summary>
/// Type converter for single-element value tuples.
/// </summary>
/// <typeparam name="T">The type of the element in the value tuple.</typeparam>
public class ValueTupleConverter<T> : TypeConverter
{
    /// <inheritdoc/>
    public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
    {
        if (sourceType == typeof(string) || sourceType == typeof(ValueTuple<T>))
            return true;

        return base.CanConvertFrom(context, sourceType);
    }

    /// <inheritdoc/>
    public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
    {
        if (destinationType == typeof(string) || destinationType == typeof(ValueTuple<T>))
            return true;

        return base.CanConvertTo(context, destinationType);
    }

    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue)
        {
            var elements = stringValue.Trim('(', ')').Split(',');
            if (elements.Length == 1)
            {
                var item = (T)TypeDescriptor.GetConverter(typeof(T)).ConvertFromString(context, culture, elements[0].Trim());
                return ValueTuple.Create(item);
            }
            throw new ArgumentException($"Cannot convert '{stringValue}' to ValueTuple<{typeof(T).Name}>.");
        }

        return base.ConvertFrom(context, culture, value);
    }

    /// <inheritdoc/>
    public override object ConvertTo(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value, Type destinationType)
    {
        if (value is ValueTuple<T> tuple)
        {
            if (destinationType == typeof(string))
            {
                var item = TypeDescriptor.GetConverter(typeof(T)).ConvertToString(context, culture, tuple.Item1);
                return $"({item})";
            }
        }

        return base.ConvertTo(context, culture, value, destinationType);
    }
}

/// <summary>
/// Type converter for single-element nullable value tuples.
/// </summary>
/// <typeparam name="T">The type of the element in the value tuple.</typeparam>
public class NullableValueTupleConverter<T> : ValueTupleConverter<T>
{
    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue && string.IsNullOrEmpty(stringValue))
            return null;
        return base.ConvertFrom(context, culture, value);
    }
}

/// <summary>
/// Type converter for two-element value tuples.
/// </summary>
/// <typeparam name="T1">The type of the first element in the value tuple.</typeparam>
/// <typeparam name="T2">The type of the second element in the value tuple.</typeparam>
public class ValueTupleConverter<T1, T2> : TypeConverter
{
    /// <inheritdoc/>
    public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
    {
        if (sourceType == typeof(string) || sourceType == typeof(ValueTuple<T1, T2>))
            return true;

        return base.CanConvertFrom(context, sourceType);
    }

    /// <inheritdoc/>
    public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
    {
        if (destinationType == typeof(string) || destinationType == typeof(ValueTuple<T1, T2>))
            return true;

        return base.CanConvertTo(context, destinationType);
    }

    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue)
        {
            var elements = stringValue.Trim('(', ')').Split(',');
            if (elements.Length == 2)
            {
                var item1 = (T1)TypeDescriptor.GetConverter(typeof(T1)).ConvertFromString(context, culture, elements[0].Trim());
                var item2 = (T2)TypeDescriptor.GetConverter(typeof(T2)).ConvertFromString(context, culture, elements[1].Trim());
                return ValueTuple.Create(item1, item2);
            }
            throw new ArgumentException($"Cannot convert '{stringValue}' to ValueTuple<{typeof(T1).Name}, {typeof(T2).Name}>.");
        }

        return base.ConvertFrom(context, culture, value);
    }

    /// <inheritdoc/>
    public override object ConvertTo(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value, Type destinationType)
    {
        if (value is ValueTuple<T1, T2> tuple)
        {
            if (destinationType == typeof(string))
            {
                var item1 = TypeDescriptor.GetConverter(typeof(T1)).ConvertToString(context, culture, tuple.Item1);
                var item2 = TypeDescriptor.GetConverter(typeof(T2)).ConvertToString(context, culture, tuple.Item2);
                return $"({item1}, {item2})";
            }
        }

        return base.ConvertTo(context, culture, value, destinationType);
    }
}

/// <summary>
/// Type converter for two-element nullable value tuples.
/// </summary>
/// <typeparam name="T1">The type of the first element in the value tuple.</typeparam>
/// <typeparam name="T2">The type of the second element in the value tuple.</typeparam>
public class NullableValueTupleConverter<T1, T2> : ValueTupleConverter<T1, T2>
{
    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue && string.IsNullOrEmpty(stringValue))
            return null;
        return base.ConvertFrom(context, culture, value);
    }
}

/// <summary>
/// Type converter for three-element value tuples.
/// </summary>
/// <typeparam name="T1">The type of the first element in the value tuple.</typeparam>
/// <typeparam name="T2">The type of the second element in the value tuple.</typeparam>
/// <typeparam name="T3">The type of the third element in the value tuple.</typeparam>
public class ValueTupleConverter<T1, T2, T3> : TypeConverter
{
    /// <inheritdoc/>
    public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
    {
        if (sourceType == typeof(string) || sourceType == typeof(ValueTuple<T1, T2, T3>))
            return true;

        return base.CanConvertFrom(context, sourceType);
    }

    /// <inheritdoc/>
    public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
    {
        if (destinationType == typeof(string) || destinationType == typeof(ValueTuple<T1, T2, T3>))
            return true;

        return base.CanConvertTo(context, destinationType);
    }

    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue)
        {
            var elements = stringValue.Trim('(', ')').Split(',');
            if (elements.Length == 3)
            {
                var item1 = (T1)TypeDescriptor.GetConverter(typeof(T1)).ConvertFromString(context, culture, elements[0].Trim());
                var item2 = (T2)TypeDescriptor.GetConverter(typeof(T2)).ConvertFromString(context, culture, elements[1].Trim());
                var item3 = (T3)TypeDescriptor.GetConverter(typeof(T3)).ConvertFromString(context, culture, elements[2].Trim());
                return ValueTuple.Create(item1, item2, item3);
            }
            throw new ArgumentException($"Cannot convert '{stringValue}' to ValueTuple<{typeof(T1).Name}, {typeof(T2).Name}, {typeof(T3).Name}>.");
        }

        return base.ConvertFrom(context, culture, value);
    }

    /// <inheritdoc/>
    public override object ConvertTo(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value, Type destinationType)
    {
        if (value is ValueTuple<T1, T2, T3> tuple)
        {
            if (destinationType == typeof(string))
            {
                var item1 = TypeDescriptor.GetConverter(typeof(T1)).ConvertToString(context, culture, tuple.Item1);
                var item2 = TypeDescriptor.GetConverter(typeof(T2)).ConvertToString(context, culture, tuple.Item2);
                var item3 = TypeDescriptor.GetConverter(typeof(T3)).ConvertToString(context, culture, tuple.Item3);
                return $"({item1}, {item2}, {item3})";
            }
        }

        return base.ConvertTo(context, culture, value, destinationType);
    }
}

/// <summary>
/// Type converter for three-element nullable value tuples.
/// </summary>
/// <typeparam name="T1">The type of the first element in the value tuple.</typeparam>
/// <typeparam name="T2">The type of the second element in the value tuple.</typeparam>
/// <typeparam name="T3">The type of the third element in the value tuple.</typeparam>
public class NullableValueTupleConverter<T1, T2, T3> : ValueTupleConverter<T1, T2, T3>
{
    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue && string.IsNullOrEmpty(stringValue))
            return null;
        return base.ConvertFrom(context, culture, value);
    }
}

/// <summary>
/// Type converter for four-element value tuples.
/// </summary>
/// <typeparam name="T1">The type of the first element in the value tuple.</typeparam>
/// <typeparam name="T2">The type of the second element in the value tuple.</typeparam>
/// <typeparam name="T3">The type of the third element in the value tuple.</typeparam>
/// <typeparam name="T4">The type of the fourth element in the value tuple.</typeparam>
public class ValueTupleConverter<T1, T2, T3, T4> : TypeConverter
{
    /// <inheritdoc/>
    public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
    {
        if (sourceType == typeof(string) || sourceType == typeof(ValueTuple<T1, T2, T3, T4>))
            return true;

        return base.CanConvertFrom(context, sourceType);
    }

    /// <inheritdoc/>
    public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
    {
        if (destinationType == typeof(string) || destinationType == typeof(ValueTuple<T1, T2, T3, T4>))
            return true;

        return base.CanConvertTo(context, destinationType);
    }

    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue)
        {
            var elements = stringValue.Trim('(', ')').Split(',');
            if (elements.Length == 3)
            {
                var item1 = (T1)TypeDescriptor.GetConverter(typeof(T1)).ConvertFromString(context, culture, elements[0].Trim());
                var item2 = (T2)TypeDescriptor.GetConverter(typeof(T2)).ConvertFromString(context, culture, elements[1].Trim());
                var item3 = (T3)TypeDescriptor.GetConverter(typeof(T3)).ConvertFromString(context, culture, elements[2].Trim());
                var item4 = (T4)TypeDescriptor.GetConverter(typeof(T4)).ConvertFromString(context, culture, elements[3].Trim());
                return ValueTuple.Create(item1, item2, item3, item4);
            }
            throw new ArgumentException($"Cannot convert '{stringValue}' to ValueTuple<{typeof(T1).Name}, {typeof(T2).Name}, {typeof(T3).Name}, {typeof(T4).Name}>.");
        }

        return base.ConvertFrom(context, culture, value);
    }

    /// <inheritdoc/>
    public override object ConvertTo(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value, Type destinationType)
    {
        if (value is ValueTuple<T1, T2, T3, T4> tuple)
        {
            if (destinationType == typeof(string))
            {
                var item1 = TypeDescriptor.GetConverter(typeof(T1)).ConvertToString(context, culture, tuple.Item1);
                var item2 = TypeDescriptor.GetConverter(typeof(T2)).ConvertToString(context, culture, tuple.Item2);
                var item3 = TypeDescriptor.GetConverter(typeof(T3)).ConvertToString(context, culture, tuple.Item3);
                var item4 = TypeDescriptor.GetConverter(typeof(T4)).ConvertToString(context, culture, tuple.Item4);
                return $"({item1}, {item2}, {item3}, {item4})";
            }
        }

        return base.ConvertTo(context, culture, value, destinationType);
    }
}

/// <summary>
/// Type converter for four-element nullable value tuples.
/// </summary>
/// <typeparam name="T1">The type of the first element in the value tuple.</typeparam>
/// <typeparam name="T2">The type of the second element in the value tuple.</typeparam>
/// <typeparam name="T3">The type of the third element in the value tuple.</typeparam>
/// <typeparam name="T4">The type of the fourth element in the value tuple.</typeparam>
public class NullableValueTupleConverter<T1, T2, T3, T4> : ValueTupleConverter<T1, T2, T3, T4>
{
    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue && string.IsNullOrEmpty(stringValue))
            return null;
        return base.ConvertFrom(context, culture, value);
    }
}

/// <summary>
/// Type converter for five-element value tuples.
/// </summary>
/// <typeparam name="T1">The type of the first element in the value tuple.</typeparam>
/// <typeparam name="T2">The type of the second element in the value tuple.</typeparam>
/// <typeparam name="T3">The type of the third element in the value tuple.</typeparam>
/// <typeparam name="T4">The type of the fourth element in the value tuple.</typeparam>
/// <typeparam name="T5">The type of the fifth element in the value tuple.</typeparam>
public class ValueTupleConverter<T1, T2, T3, T4, T5> : TypeConverter
{
    /// <inheritdoc/>
    public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
    {
        if (sourceType == typeof(string) || sourceType == typeof(ValueTuple<T1, T2, T3, T4, T5>))
            return true;

        return base.CanConvertFrom(context, sourceType);
    }

    /// <inheritdoc/>
    public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
    {
        if (destinationType == typeof(string) || destinationType == typeof(ValueTuple<T1, T2, T3, T4, T5>))
            return true;

        return base.CanConvertTo(context, destinationType);
    }

    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue)
        {
            var elements = stringValue.Trim('(', ')').Split(',');
            if (elements.Length == 3)
            {
                var item1 = (T1)TypeDescriptor.GetConverter(typeof(T1)).ConvertFromString(context, culture, elements[0].Trim());
                var item2 = (T2)TypeDescriptor.GetConverter(typeof(T2)).ConvertFromString(context, culture, elements[1].Trim());
                var item3 = (T3)TypeDescriptor.GetConverter(typeof(T3)).ConvertFromString(context, culture, elements[2].Trim());
                var item4 = (T4)TypeDescriptor.GetConverter(typeof(T4)).ConvertFromString(context, culture, elements[3].Trim());
                var item5 = (T5)TypeDescriptor.GetConverter(typeof(T5)).ConvertFromString(context, culture, elements[4].Trim());
                return ValueTuple.Create(item1, item2, item3, item4, item5);
            }
            throw new ArgumentException($"Cannot convert '{stringValue}' to ValueTuple<{typeof(T1).Name}, {typeof(T2).Name}, {typeof(T3).Name}, {typeof(T4).Name}, {typeof(T5).Name}>.");
        }

        return base.ConvertFrom(context, culture, value);
    }

    /// <inheritdoc/>
    public override object ConvertTo(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value, Type destinationType)
    {
        if (value is ValueTuple<T1, T2, T3, T4, T5> tuple)
        {
            if (destinationType == typeof(string))
            {
                var item1 = TypeDescriptor.GetConverter(typeof(T1)).ConvertToString(context, culture, tuple.Item1);
                var item2 = TypeDescriptor.GetConverter(typeof(T2)).ConvertToString(context, culture, tuple.Item2);
                var item3 = TypeDescriptor.GetConverter(typeof(T3)).ConvertToString(context, culture, tuple.Item3);
                var item4 = TypeDescriptor.GetConverter(typeof(T4)).ConvertToString(context, culture, tuple.Item4);
                var item5 = TypeDescriptor.GetConverter(typeof(T5)).ConvertToString(context, culture, tuple.Item5);
                return $"({item1}, {item2}, {item3}, {item4}, {item5})";
            }
        }

        return base.ConvertTo(context, culture, value, destinationType);
    }
}

/// <summary>
/// Type converter for five-element nullable value tuples.
/// </summary>
/// <typeparam name="T1">The type of the first element in the value tuple.</typeparam>
/// <typeparam name="T2">The type of the second element in the value tuple.</typeparam>
/// <typeparam name="T3">The type of the third element in the value tuple.</typeparam>
/// <typeparam name="T4">The type of the fourth element in the value tuple.</typeparam>
/// <typeparam name="T5">The type of the fifth element in the value tuple.</typeparam>
public class NullableValueTupleConverter<T1, T2, T3, T4, T5> : ValueTupleConverter<T1, T2, T3, T4, T5>
{
    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue && string.IsNullOrEmpty(stringValue))
            return null;
        return base.ConvertFrom(context, culture, value);
    }
}

/// <summary>
/// Type converter for six-element value tuples.
/// </summary>
/// <typeparam name="T1">The type of the first element in the value tuple.</typeparam>
/// <typeparam name="T2">The type of the second element in the value tuple.</typeparam>
/// <typeparam name="T3">The type of the third element in the value tuple.</typeparam>
/// <typeparam name="T4">The type of the fourth element in the value tuple.</typeparam>
/// <typeparam name="T5">The type of the fifth element in the value tuple.</typeparam>
/// <typeparam name="T6">The type of the sixth element in the value tuple.</typeparam>
public class ValueTupleConverter<T1, T2, T3, T4, T5, T6> : TypeConverter
{
    /// <inheritdoc/>
    public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
    {
        if (sourceType == typeof(string) || sourceType == typeof(ValueTuple<T1, T2, T3, T4, T5, T6>))
            return true;

        return base.CanConvertFrom(context, sourceType);
    }

    /// <inheritdoc/>
    public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
    {
        if (destinationType == typeof(string) || destinationType == typeof(ValueTuple<T1, T2, T3, T4, T5, T6>))
            return true;

        return base.CanConvertTo(context, destinationType);
    }

    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue)
        {
            var elements = stringValue.Trim('(', ')').Split(',');
            if (elements.Length == 3)
            {
                var item1 = (T1)TypeDescriptor.GetConverter(typeof(T1)).ConvertFromString(context, culture, elements[0].Trim());
                var item2 = (T2)TypeDescriptor.GetConverter(typeof(T2)).ConvertFromString(context, culture, elements[1].Trim());
                var item3 = (T3)TypeDescriptor.GetConverter(typeof(T3)).ConvertFromString(context, culture, elements[2].Trim());
                var item4 = (T4)TypeDescriptor.GetConverter(typeof(T4)).ConvertFromString(context, culture, elements[3].Trim());
                var item5 = (T5)TypeDescriptor.GetConverter(typeof(T5)).ConvertFromString(context, culture, elements[4].Trim());
                var item6 = (T6)TypeDescriptor.GetConverter(typeof(T6)).ConvertFromString(context, culture, elements[5].Trim());
                return ValueTuple.Create(item1, item2, item3, item4, item5, item6);
            }
            throw new ArgumentException($"Cannot convert '{stringValue}' to ValueTuple<{typeof(T1).Name}, {typeof(T2).Name}, {typeof(T3).Name}, {typeof(T4).Name}, {typeof(T5).Name}, {typeof(T6).Name}>.");
        }

        return base.ConvertFrom(context, culture, value);
    }

    /// <inheritdoc/>
    public override object ConvertTo(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value, Type destinationType)
    {
        if (value is ValueTuple<T1, T2, T3, T4, T5, T6> tuple)
        {
            if (destinationType == typeof(string))
            {
                var item1 = TypeDescriptor.GetConverter(typeof(T1)).ConvertToString(context, culture, tuple.Item1);
                var item2 = TypeDescriptor.GetConverter(typeof(T2)).ConvertToString(context, culture, tuple.Item2);
                var item3 = TypeDescriptor.GetConverter(typeof(T3)).ConvertToString(context, culture, tuple.Item3);
                var item4 = TypeDescriptor.GetConverter(typeof(T4)).ConvertToString(context, culture, tuple.Item4);
                var item5 = TypeDescriptor.GetConverter(typeof(T5)).ConvertToString(context, culture, tuple.Item5);
                var item6 = TypeDescriptor.GetConverter(typeof(T6)).ConvertToString(context, culture, tuple.Item6);
                return $"({item1}, {item2}, {item3}, {item4}, {item5}, {item6})";
            }
        }

        return base.ConvertTo(context, culture, value, destinationType);
    }
}

/// <summary>
/// Type converter for six-element nullable value tuples.
/// </summary>
/// <typeparam name="T1">The type of the first element in the value tuple.</typeparam>
/// <typeparam name="T2">The type of the second element in the value tuple.</typeparam>
/// <typeparam name="T3">The type of the third element in the value tuple.</typeparam>
/// <typeparam name="T4">The type of the fourth element in the value tuple.</typeparam>
/// <typeparam name="T5">The type of the fifth element in the value tuple.</typeparam>
/// <typeparam name="T6">The type of the sixth element in the value tuple.</typeparam>
public class NullableValueTupleConverter<T1, T2, T3, T4, T5, T6> : ValueTupleConverter<T1, T2, T3, T4, T5, T6>
{
    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue && string.IsNullOrEmpty(stringValue))
            return null;
        return base.ConvertFrom(context, culture, value);
    }
}

/// <summary>
/// Type converter for seven-element value tuples.
/// </summary>
/// <typeparam name="T1">The type of the first element in the value tuple.</typeparam>
/// <typeparam name="T2">The type of the second element in the value tuple.</typeparam>
/// <typeparam name="T3">The type of the third element in the value tuple.</typeparam>
/// <typeparam name="T4">The type of the fourth element in the value tuple.</typeparam>
/// <typeparam name="T5">The type of the fifth element in the value tuple.</typeparam>
/// <typeparam name="T6">The type of the sixth element in the value tuple.</typeparam>
/// <typeparam name="T7">The type of the seventh element in the value tuple.</typeparam>
public class ValueTupleConverter<T1, T2, T3, T4, T5, T6, T7> : TypeConverter
{
    /// <inheritdoc/>
    public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
    {
        if (sourceType == typeof(string) || sourceType == typeof(ValueTuple<T1, T2, T3, T4, T5, T6, T7>))
            return true;

        return base.CanConvertFrom(context, sourceType);
    }

    /// <inheritdoc/>
    public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
    {
        if (destinationType == typeof(string) || destinationType == typeof(ValueTuple<T1, T2, T3, T4, T5, T6, T7>))
            return true;

        return base.CanConvertTo(context, destinationType);
    }

    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue)
        {
            var elements = stringValue.Trim('(', ')').Split(',');
            if (elements.Length == 3)
            {
                var item1 = (T1)TypeDescriptor.GetConverter(typeof(T1)).ConvertFromString(context, culture, elements[0].Trim());
                var item2 = (T2)TypeDescriptor.GetConverter(typeof(T2)).ConvertFromString(context, culture, elements[1].Trim());
                var item3 = (T3)TypeDescriptor.GetConverter(typeof(T3)).ConvertFromString(context, culture, elements[2].Trim());
                var item4 = (T4)TypeDescriptor.GetConverter(typeof(T4)).ConvertFromString(context, culture, elements[3].Trim());
                var item5 = (T5)TypeDescriptor.GetConverter(typeof(T5)).ConvertFromString(context, culture, elements[4].Trim());
                var item6 = (T6)TypeDescriptor.GetConverter(typeof(T6)).ConvertFromString(context, culture, elements[5].Trim());
                var item7 = (T7)TypeDescriptor.GetConverter(typeof(T7)).ConvertFromString(context, culture, elements[6].Trim());
                return ValueTuple.Create(item1, item2, item3, item4, item5, item6, item7);
            }
            throw new ArgumentException($"Cannot convert '{stringValue}' to ValueTuple<{typeof(T1).Name}, {typeof(T2).Name}, {typeof(T3).Name}, {typeof(T4).Name}, {typeof(T5).Name}, {typeof(T6).Name}, {typeof(T7).Name}>.");
        }

        return base.ConvertFrom(context, culture, value);
    }

    /// <inheritdoc/>
    public override object ConvertTo(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value, Type destinationType)
    {
        if (value is ValueTuple<T1, T2, T3, T4, T5, T6, T7> tuple)
        {
            if (destinationType == typeof(string))
            {
                var item1 = TypeDescriptor.GetConverter(typeof(T1)).ConvertToString(context, culture, tuple.Item1);
                var item2 = TypeDescriptor.GetConverter(typeof(T2)).ConvertToString(context, culture, tuple.Item2);
                var item3 = TypeDescriptor.GetConverter(typeof(T3)).ConvertToString(context, culture, tuple.Item3);
                var item4 = TypeDescriptor.GetConverter(typeof(T4)).ConvertToString(context, culture, tuple.Item4);
                var item5 = TypeDescriptor.GetConverter(typeof(T5)).ConvertToString(context, culture, tuple.Item5);
                var item6 = TypeDescriptor.GetConverter(typeof(T6)).ConvertToString(context, culture, tuple.Item6);
                var item7 = TypeDescriptor.GetConverter(typeof(T7)).ConvertToString(context, culture, tuple.Item7);
                return $"({item1}, {item2}, {item3}, {item4}, {item5}, {item6}, {item7})";
            }
        }

        return base.ConvertTo(context, culture, value, destinationType);
    }
}

/// <summary>
/// Type converter for seven-element nullable value tuples.
/// </summary>
/// <typeparam name="T1">The type of the first element in the value tuple.</typeparam>
/// <typeparam name="T2">The type of the second element in the value tuple.</typeparam>
/// <typeparam name="T3">The type of the third element in the value tuple.</typeparam>
/// <typeparam name="T4">The type of the fourth element in the value tuple.</typeparam>
/// <typeparam name="T5">The type of the fifth element in the value tuple.</typeparam>
/// <typeparam name="T6">The type of the sixth element in the value tuple.</typeparam>
/// <typeparam name="T7">The type of the seventh element in the value tuple.</typeparam>
public class NullableValueTupleConverter<T1, T2, T3, T4, T5, T6, T7> : ValueTupleConverter<T1, T2, T3, T4, T5, T6, T7>
{
    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
    {
        if (value is string stringValue && string.IsNullOrEmpty(stringValue))
            return null;
        return base.ConvertFrom(context, culture, value);
    }
}
