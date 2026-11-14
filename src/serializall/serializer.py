# -*- coding: utf-8 -*-
"""
Created the 20/10/2023

@author: Sebastien Weber
"""
import numbers
from typing import Tuple, List, Union
from collections import OrderedDict
from datetime import date, datetime, time
from pathlib import Path, PosixPath, PurePath, WindowsPath

import numpy as np

from . import utils
from .factory import SerializableFactory, SERIALIZABLE, SerializableBase

ser_factory = SerializableFactory()


class NoneSerializeDeserialize(SerializableBase):
    @staticmethod
    def serialize(obj: None) -> bytes:  # type: ignore[override]
        return b""

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[None, bytes]:  # type: ignore[override]
        return None, bytes_str


class StringSerializeDeserialize(SerializableBase):

    @staticmethod
    def serialize(string: str) -> bytes:
        """ Convert a string into a bytes message together with the info to convert it back

        Parameters
        ----------
        string: str

        Returns
        -------
        bytes: the total bytes message to serialize the string
        """
        bytes_string = b''
        cmd_bytes, cmd_length_bytes = utils.str_len_to_bytes(string)
        bytes_string += cmd_length_bytes
        bytes_string += cmd_bytes
        return bytes_string

    @staticmethod
    def deserialize(bytes_str) -> Tuple[str, bytes]:
        """Convert bytes into a str object

        Convert first the fourth first bytes into an int encoding the length of the string to decode

        Returns
        -------
        str: the decoded string
        bytes: the remaining bytes string if any
        """
        string_len, remaining_bytes = utils.get_int_from_bytes(bytes_str)
        str_bytes,  remaining_bytes = utils.split_nbytes(remaining_bytes, string_len)
        str_obj = utils.bytes_to_string(str_bytes)
        return str_obj, remaining_bytes


class BytesSerializeDeserialize(SerializableBase):
    @staticmethod
    def serialize(some_bytes: bytes) -> bytes:
        bytes_string = b''
        bytes_string += utils.int_to_bytes(len(some_bytes))
        bytes_string += some_bytes
        return bytes_string

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[bytes, bytes]:
        bytes_len, remaining_bytes = utils.get_int_from_bytes(bytes_str)
        bytes_str, remaining_bytes = utils.split_nbytes(remaining_bytes, bytes_len)
        return bytes_str, remaining_bytes


class ScalarSerializeDeserialize(SerializableBase):
    @staticmethod
    def serialize(scalar: complex) -> bytes:
        """ Convert a scalar into a bytes message together with the info to convert it back

        Parameters
        ----------
        scalar: A python number (complex or subtypes like float and int)

        Returns
        -------
        bytes: the total bytes message to serialize the scalar
        """
        if not isinstance(scalar, numbers.Number):
            # type hint is complex, instance comparison Number
            raise TypeError(f'{scalar} should be an integer or a float, not a {type(scalar)}')
        scalar_array = np.array([scalar])
        data_type = scalar_array.dtype.descr[0][1]
        data_bytes = scalar_array.tobytes()

        bytes_string = b''
        bytes_string += StringSerializeDeserialize.serialize(data_type)
        bytes_string += utils.int_to_bytes(len(data_bytes))
        bytes_string += data_bytes
        return bytes_string

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[complex, bytes]:
        """Convert bytes into a python object of type (float, int, complex or boolean)

        Get first the data type from a string deserialization, then the data length and finally convert this
        length of bytes into an object of type (float, int, complex or boolean)

        Returns
        -------
        numbers.Number: the decoded number
        bytes: the remaining bytes string if any
        """
        data_type, remaining_bytes = StringSerializeDeserialize.deserialize(bytes_str)
        data_len, remaining_bytes = utils.get_int_from_bytes(remaining_bytes)
        number_bytes, remaining_bytes = utils.split_nbytes(remaining_bytes, data_len)
        number = np.frombuffer(number_bytes, dtype=data_type)[0]
        if 'f' in data_type:
            number = float(number)  # because one get numpy float type
        elif 'i' in data_type:
            number = int(number)  # because one get numpy int type
        elif 'c' in data_type:
            number = complex(number)  # because one get numpy complex type
        elif 'b' in data_type:
            number = bool(number)  # because one get numpy complex type
        return number, remaining_bytes


class NdArraySerializeDeserialize(SerializableBase):

    @staticmethod
    def serialize(array: np.ndarray) -> bytes:
        """ Convert a ndarray into a bytes message together with the info to convert it back

        Parameters
        ----------
        array: np.ndarray

        Returns
        -------
        bytes: the total bytes message to serialize the scalar

        Notes
        -----

        The bytes sequence is constructed as:

        * get data type as a string
        * reshape array as 1D array and get the array dimensionality (len of array's shape)
        * convert Data array as bytes
        * serialize data type
        * serialize data length
        * serialize data shape length
        * serialize all values of the shape as integers converted to bytes
        * serialize array as bytes
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f'{array} should be an numpy array, not a {type(array)}')
        array_type = array.dtype.descr[0][1]
        array_shape = array.shape

        array = array.reshape(array.size)
        array_bytes = array.tobytes()
        bytes_string = b''
        bytes_string += StringSerializeDeserialize.serialize(array_type)
        bytes_string += utils.int_to_bytes(len(array_bytes))
        bytes_string += utils.int_to_bytes(len(array_shape))
        for shape_elt in array_shape:
            bytes_string += utils.int_to_bytes(shape_elt)
        bytes_string += array_bytes
        return bytes_string

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[np.ndarray, bytes]:
        """Convert bytes into a numpy ndarray object

        Convert the first bytes into a ndarray reading first information about the array's data

        Returns
        -------
        ndarray: the decoded numpy array
        bytes: the remaining bytes string if any
        """
        ndarray_type, remaining_bytes = StringSerializeDeserialize.deserialize(bytes_str)
        ndarray_len, remaining_bytes = utils.get_int_from_bytes(remaining_bytes)
        shape_len, remaining_bytes = utils.get_int_from_bytes(remaining_bytes)
        shape = []
        for ind in range(shape_len):
            shape_elt, remaining_bytes = utils.get_int_from_bytes(remaining_bytes)
            shape.append(shape_elt)

        ndarray_bytes, remaining_bytes = utils.split_nbytes(remaining_bytes, ndarray_len)
        ndarray = np.frombuffer(ndarray_bytes, dtype=ndarray_type)
        ndarray = ndarray.reshape(tuple(shape))
        ndarray = np.atleast_1d(ndarray)  # remove singleton dimensions
        return ndarray, remaining_bytes


class ListSerializeDeserialize(SerializableBase):
    @staticmethod
    def serialize(list_object: List) -> bytes:
        """ Convert a list of objects into a bytes message together with the info to convert it back

        Parameters
        ----------
        list_object: list
            the list could contain whatever objects are registered in the SerializableFactory

        Returns
        -------
        bytes: the total bytes message to serialize the list of objects

        Notes
        -----

        The bytes sequence is constructed as:
        * the length of the list

        Then for each object:
        * use the serialization method adapted to each object in the list
        """
        if not isinstance(list_object, list):
            raise TypeError(f'{list_object} should be a list, not a {type(list_object)}')

        bytes_string = b''
        bytes_string += utils.int_to_bytes(len(list_object))
        for obj in list_object:
            bytes_string += ser_factory.get_apply_serializer(obj)
        return bytes_string

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[List[SERIALIZABLE], bytes]:
        """Convert bytes into a list of objects

        Convert the first bytes into a list reading first information about the list elt types, length ...

        Returns
        -------
        list: the decoded list
        bytes: the remaining bytes string if any
        """
        list_obj = []
        list_len, remaining_bytes = utils.get_int_from_bytes(bytes_str)

        for ind in range(list_len):
            obj, remaining_bytes = ser_factory.get_apply_deserializer(remaining_bytes,
                                                                      only_object=False)
            list_obj.append(obj)
        return list_obj, remaining_bytes


class TupleSerializeDeserialize(SerializableBase):
    @staticmethod
    def serialize(tuple_object: Tuple[SERIALIZABLE, ...]) -> bytes:
        """ Convert a tuple of objects into a bytes message together with the info to convert it back

        Parameters
        ----------
        tuple_object: tuple
            the tuple could contain whatever objects are registered in the SerializableFactory

        Returns
        -------
        bytes: the total bytes message to serialize the tuple of objects

        Notes
        -----

        The bytes sequence is constructed as:
        * the length of the tuple

        Then for each object:
        * use the serialization method adapted to each object in the tuple
        """
        if not isinstance(tuple_object, tuple):
            raise TypeError(f'{tuple_object} should be a tuple, not a {type(tuple_object)}')
        return ListSerializeDeserialize().serialize(list(tuple_object))

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[Tuple[SERIALIZABLE, ...], bytes]:
        """Convert bytes into a tuple of objects

        Convert the first bytes into a tuple reading first information about the tuple elt types, length ...

        Returns
        -------
        tuple: the decoded tuple
        bytes: the remaining bytes string if any
        """
        list_object, remaining_bytes = ListSerializeDeserialize().deserialize(bytes_str)
        return tuple(list_object), remaining_bytes


class DictSerializeDeserialize(SerializableBase):
    @staticmethod
    def serialize(dict_object: dict[SERIALIZABLE, SERIALIZABLE]) -> bytes:
        """ Convert a dictionnary of objects into a bytes message together with the info to convert it back

        Parameters
        ----------
        dict_object: dict
            the dict could contain whatever objects are registered in the SerializableFactory

        Returns
        -------
        bytes: the total bytes message to serialize the list of objects

        Notes
        -----

        The bytes sequence is constructed as:
        * the list of keys of the dict

        Then for each key:
        * use the serialization method adapted to the object inferred from the key
        """
        if not isinstance(dict_object, dict):
            raise TypeError(f'{dict_object} should be a dict, not a {type(dict_object)}')

        bytes_string = b''
        keys_list = list(dict_object.keys())
        bytes_string += ser_factory.get_apply_serializer(keys_list)
        for key in keys_list:
            bytes_string += ser_factory.get_apply_serializer(dict_object[key])
        return bytes_string

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[dict[str, SERIALIZABLE], bytes]:
        """Convert bytes into a dictionary of serializable objects

        Convert the first bytes into a dict reading first information about the key elts of the dictionnary then
        the underlying objects ...

        Returns
        -------
        dict: the decoded dictionary
        bytes: the remaining bytes string if any
        """
        dict_object = {}
        keys_list, remaining_bytes = ser_factory.get_apply_deserializer(bytes_str,
                                                                        only_object=False)

        for key in keys_list:
            obj, remaining_bytes = ser_factory.get_apply_deserializer(remaining_bytes,
                                                                      only_object=False)
            dict_object[key] = obj
        return dict_object, remaining_bytes


class OrderedDictSerializeDeserialize(SerializableBase):
    @staticmethod
    def serialize(ordered_dict: OrderedDict) -> bytes:
        """
        Convert an OrderedDict object into a bytes message together with the info to convert it back

        Parameters
        ----------
        ordered_dict: OrderedDict
            the ordered dict to convert

        Returns
        -------
        bytes: the total bytes message to serialize the OrderedDict object
        """
        if not isinstance(ordered_dict, OrderedDict):
            raise TypeError(f'{ordered_dict} should be an OrderedDict, not a {type(ordered_dict)}')

        dict_list = list(ordered_dict.items())
        return ListSerializeDeserialize.serialize(dict_list)

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[OrderedDict, bytes]:
        """
        Convert bytes into an OrderedDict object

        Returns
        -------
        OrderedDict: the decoded ordered dict
        bytes: the remaining bytes string if any
        """
        dict_list, remaining_bytes = ListSerializeDeserialize.deserialize(bytes_str)
        return OrderedDict(dict_list), remaining_bytes


class NumpyScalarSerializeDeserialize(SerializableBase):
    """Numpy scalars (np.float32, np.int8, np.bool_...): same bytes as the python scalars, but the
    deserialized object is a numpy scalar of the same dtype.

    The python-compatible types np.float64, np.int64, np.complex128 and np.bool_ are deserialized as the
    python float, int, complex and bool: same value, same behaviour, and what libraries expect (json...).
    """
    @staticmethod
    def serialize(scalar: np.generic) -> bytes:
        """ Convert a numpy scalar into a bytes message: dtype, length, raw bytes """
        array = np.array([scalar])
        data_bytes = array.tobytes()
        bytes_string = StringSerializeDeserialize.serialize(array.dtype.descr[0][1])
        bytes_string += utils.int_to_bytes(len(data_bytes))
        bytes_string += data_bytes
        return bytes_string

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[np.generic, bytes]:
        """ Convert bytes into a numpy scalar of the serialized dtype

        Returns
        -------
        np.generic: the decoded scalar
        bytes: the remaining bytes string if any
        """
        data_type, remaining_bytes = StringSerializeDeserialize.deserialize(bytes_str)
        data_len, remaining_bytes = utils.get_int_from_bytes(remaining_bytes)
        number_bytes, remaining_bytes = utils.split_nbytes(remaining_bytes, data_len)
        return np.frombuffer(number_bytes, dtype=data_type)[0], remaining_bytes


class SetSerializeDeserialize(SerializableBase):
    """set and frozenset, serialized as a list of their items (in no particular order)"""
    @staticmethod
    def serialize(set_object: Union[set, frozenset]) -> bytes:
        return ListSerializeDeserialize.serialize(list(set_object))

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[set, bytes]:
        items, remaining_bytes = ListSerializeDeserialize.deserialize(bytes_str)
        return set(items), remaining_bytes


class FrozenSetSerializeDeserialize(SetSerializeDeserialize):
    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[frozenset, bytes]:
        items, remaining_bytes = ListSerializeDeserialize.deserialize(bytes_str)
        return frozenset(items), remaining_bytes


class PathSerializeDeserialize(SerializableBase):
    """pathlib paths, serialized as a string and rebuilt as a ``pathlib.Path`` of the local platform"""
    @staticmethod
    def serialize(path: PurePath) -> bytes:
        return StringSerializeDeserialize.serialize(str(path))

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[Path, bytes]:
        path_str, remaining_bytes = StringSerializeDeserialize.deserialize(bytes_str)
        return Path(path_str), remaining_bytes


class DatetimeSerializeDeserialize(SerializableBase):
    """datetime, serialized as an ISO 8601 string (timezone and microseconds are kept)"""
    @staticmethod
    def serialize(dt: datetime) -> bytes:
        return StringSerializeDeserialize.serialize(dt.isoformat())

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[datetime, bytes]:
        iso, remaining_bytes = StringSerializeDeserialize.deserialize(bytes_str)
        return datetime.fromisoformat(iso), remaining_bytes


class DateSerializeDeserialize(SerializableBase):
    """date, serialized as an ISO 8601 string"""
    @staticmethod
    def serialize(d: date) -> bytes:
        return StringSerializeDeserialize.serialize(d.isoformat())

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[date, bytes]:
        iso, remaining_bytes = StringSerializeDeserialize.deserialize(bytes_str)
        return date.fromisoformat(iso), remaining_bytes


class TimeSerializeDeserialize(SerializableBase):
    """time, serialized as an ISO 8601 string"""
    @staticmethod
    def serialize(t: time) -> bytes:
        return StringSerializeDeserialize.serialize(t.isoformat())

    @staticmethod
    def deserialize(bytes_str: bytes) -> Tuple[time, bytes]:
        iso, remaining_bytes = StringSerializeDeserialize.deserialize(bytes_str)
        return time.fromisoformat(iso), remaining_bytes


ser_factory.register_from_type(
    type(None), NoneSerializeDeserialize.serialize, NoneSerializeDeserialize.deserialize
)
ser_factory.register_from_type(bytes,
                                       BytesSerializeDeserialize.serialize,
                                       BytesSerializeDeserialize.deserialize)
ser_factory.register_from_type(str, StringSerializeDeserialize.serialize,
                                       StringSerializeDeserialize.deserialize)
ser_factory.register_from_type(int, ScalarSerializeDeserialize.serialize,
                                       ScalarSerializeDeserialize.deserialize)
ser_factory.register_from_type(float, ScalarSerializeDeserialize.serialize,
                                       ScalarSerializeDeserialize.deserialize)
ser_factory.register_from_obj(1 + 1j, ScalarSerializeDeserialize.serialize,
                                      ScalarSerializeDeserialize.deserialize)
ser_factory.register_from_type(bool, ScalarSerializeDeserialize.serialize,
                                       ScalarSerializeDeserialize.deserialize)
ser_factory.register_from_obj(np.array([0, 1]),
                                      NdArraySerializeDeserialize.serialize,
                                      NdArraySerializeDeserialize.deserialize)
ser_factory.register_from_type(list,
                               ListSerializeDeserialize.serialize,
                               ListSerializeDeserialize.deserialize)

ser_factory.register_from_type(tuple,
                               TupleSerializeDeserialize.serialize,
                               TupleSerializeDeserialize.deserialize)

ser_factory.register_from_type(dict,
                               DictSerializeDeserialize.serialize,
                               DictSerializeDeserialize.deserialize)

ser_factory.register_from_type(OrderedDict,
                               OrderedDictSerializeDeserialize.serialize,
                               OrderedDictSerializeDeserialize.deserialize)

# numpy scalars: the type name written in the stream is the numpy class name (float32, int8...)
for _np_type in (np.float16, np.float32, np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32,
                 np.uint64, np.complex64):
    ser_factory.register_from_type(_np_type,
                                   NumpyScalarSerializeDeserialize.serialize,
                                   NumpyScalarSerializeDeserialize.deserialize)
# the numpy scalars having an exact python equivalent are deserialized as the python type
for _np_type in (np.float64, np.int64, np.complex128):
    ser_factory.register_from_type(_np_type,
                                   NumpyScalarSerializeDeserialize.serialize,
                                   ScalarSerializeDeserialize.deserialize)
# np.bool_ is named 'bool' in the stream and is therefore deserialized as the python bool
ser_factory.register_from_type(np.bool_,
                               NumpyScalarSerializeDeserialize.serialize,
                               ScalarSerializeDeserialize.deserialize)

ser_factory.register_from_type(set, SetSerializeDeserialize.serialize, SetSerializeDeserialize.deserialize)
ser_factory.register_from_type(frozenset, FrozenSetSerializeDeserialize.serialize,
                               FrozenSetSerializeDeserialize.deserialize)
for _path_type in (PosixPath, WindowsPath):
    ser_factory.register_from_type(_path_type, PathSerializeDeserialize.serialize,
                                   PathSerializeDeserialize.deserialize)
ser_factory.register_from_type(datetime, DatetimeSerializeDeserialize.serialize,
                               DatetimeSerializeDeserialize.deserialize)
ser_factory.register_from_type(date, DateSerializeDeserialize.serialize, DateSerializeDeserialize.deserialize)
ser_factory.register_from_type(time, TimeSerializeDeserialize.serialize, TimeSerializeDeserialize.deserialize)
