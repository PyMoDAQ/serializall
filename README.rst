SERIALIZALL
###########



Serializall is meant to serialize any python objects. It incorporates python built-in objects serialization and
can also serialize numpy arrays.

It is based on a Factory pattern allowing to register objects that can be serialized. To do so objects should inherit
a MixIn base class exposing a mandatory interface with two methods namely *serialize* and *deserialize* implementing the
serialization using builtins (see example).

It is currently used in projects such as PyMoDAQ and pyqtgraph.


Compatibility
+++++++++++++

+-------------+-------------+---------------+
|             | Linux       | Windows       |
+=============+=============+===============+
| Python 3.9  | |39-linux|  | |39-windows|  |
+-------------+-------------+---------------+
| Python 3.10 | |310-linux| | |310-windows| |
+-------------+-------------+---------------+
| Python 3.11 | |311-linux| | |311-windows| |
+-------------+-------------+---------------+
| Python 3.12 | |312-linux| | |312-windows| |
+-------------+-------------+---------------+





.. |39-linux| image:: https://raw.githubusercontent.com/PyMoDAQ/serializall/main/tests_Linux_3.9.svg
    :target: https://github.com/PyMoDAQ/serializall/actions/workflows/tests.yml

.. |310-linux| image:: https://raw.githubusercontent.com/PyMoDAQ/serializall/main/tests_Linux_3.10.svg
    :target: https://github.com/PyMoDAQ/serializall/actions/workflows/tests.yml

.. |311-linux| image:: https://raw.githubusercontent.com/PyMoDAQ/serializall/main/tests_Linux_3.11.svg
    :target: https://github.com/PyMoDAQ/serializall/actions/workflows/tests.yml

.. |312-linux| image:: https://raw.githubusercontent.com/PyMoDAQ/serializall/main/tests_Linux_3.12.svg
    :target: https://github.com/PyMoDAQ/serializall/actions/workflows/tests.yml

.. |39-windows| image:: https://raw.githubusercontent.com/PyMoDAQ/serializall/main/tests_Windows_3.9.svg
    :target: https://github.com/PyMoDAQ/serializall/actions/workflows/tests.yml

.. |310-windows| image:: https://raw.githubusercontent.com/PyMoDAQ/serializall/main/tests_Windows_3.10.svg
    :target: https://github.com/PyMoDAQ/serializall/actions/workflows/tests.yml

.. |311-windows| image:: https://raw.githubusercontent.com/PyMoDAQ/serializall/main/tests_Windows_3.11.svg
    :target: https://github.com/PyMoDAQ/serializall/actions/workflows/tests.yml

.. |312-windows| image:: https://raw.githubusercontent.com/PyMoDAQ/serializall/main/tests_Windows_3.12.svg
    :target: https://github.com/PyMoDAQ/serializall/actions/workflows/tests.yml

Usage
+++++

The serialization is easy to perform using the SerializableFactory object and its two API methods:

* ``get_apply_serializer``
* ``get_apply_deserializer``

Example with a ``string`` object:
---------------------------------

.. code-block::

    >>> from serializall.factory import SerializableFactory

    >>> ser_factory = SerializableFactory()

    >>> astring = 'a string to serialize'

    >>> a_serialized_string = ser_factory.get_apply_serializer(astring)

    >>> print(a_serialized_string)
    b'\x00\x00\x00\x03str\x00\x00\x00\x15a string to serialize'

    >>> print(ser_factory.get_apply_deserializer(a_serialized_string))
    'a string to serialize'

From this example, one see (with good eyes) the mechanism used in the serialization. First the object type is serialized
as a string passing first the length of the string on four bytes (here 3), then the binary string itself (here the *str*
characters), followed by the length of the object on four bytes (here 15 characters) character of the
object: **a string to serialize** and finally the object itself

Example with a ``list`` object:
-------------------------------

List can contain any builtins or objects registered in the SerializableFactory

.. code-block::

    >>> from serializall.factory import SerializableFactory
    >>> import numpy as np

    >>> ser_factory = SerializableFactory()

    >>> a_list = ['hjk', 23, 34.7, np.array([1, 2, 3])]

    >>> a_serialized_list = ser_factory.get_apply_serializer(a_list)

    >>> print(a_serialized_list)
    b'\x00\x00\x00\x04list\x00\x00\x00\x04\x00\x00\x00\x03str\x00\x00\x00\x03hjk\x00\x00\x00\x03int\x00\x00\x00\x03<i8
    \x00\x00\x00\x08\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05float\x00\x00\x00\x03<f8\x00\x00\x00\x08\x9a\x99\x99
    \x99\x99YA@\x00\x00\x00\x07ndarray\x00\x00\x00\x03<i8\x00\x00\x00\x18\x00\x00\x00\x01\x00\x00\x00\x03\x01\x00\x00\x00
    \x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00'


    >>> print(ser_factory.get_apply_deserializer(a_serialized_list))
    ['hjk', 23, 34.7, array([1, 2, 3])]

From this example, one see (with good eyes) the mechanism used in the serialization. First the object type is serialized
as a string passing first the length of the string on four bytes, then the binary string (here the *list* characters),
followed by the elements of the list: a **str** (hjk characters), an **int** (serialized itself using numpy buffering),
then a **float** and finally a **ndarray**. During the deserialization steps, the type of the next element to be
deserialized will be inferred from the string characters and the right deserialization method fetch from the
factory using the retrieved type as a key

By systemizing this approach: length of the type, type as a string, length of the serialized object itself
and finally serialized object, the mechanism is stable and the factory will work seamlessly independently of the
particular object serialization (handles in both ``serialize`` and ``deserialize`` methods (see below).


Implementation example:
+++++++++++++++++++++++

Let's say we have a custom object defined as below, a kind of DataClass only storing two attributes:

.. code-block::

    class MyObject():
        def __init__(self, a: int, b: str):
            self.a = a
            self.b = b

The first thing to do is to force the SerializableBase interface using inheritance and to register the object in the
factory using the decorator ``SerializableFactory.register_decorator`` on the class itself.

.. code-block::

    from serializall.factory import SerializableFactory, SerializableBase

    @SerializableFactory.register_decorator()
    class MyObject(SerializableBase):
        def __init__(self, a: int, b: str):
            super().__init()
            self.a = a
            self.b = b

        @staticmethod
        def serialize(param: 'MyObject') -> bytes:
            """ Implement the MyObject type serialization """
            pass

        @staticmethod
        def deserialize(bytes_str: bytes) -> tuple['MyObject', bytes]:
            """ Implements deserialization into a MyObject object from bytes """
            pass

As you can see the two methods are not completely symmetric as the ``deserialize`` method returns a tuple containing
an instance ot the object and eventual remaining bytes. This is important in the case where your object has been, for
instance, part of a list object. The remaining bytes will contains the eventual next objects of the list to deserialize.

It is then up to you to do the actual serialization/deserialization using the builtins methods already present in the
factory. For instance:

.. code-block::

    from serializall.factory import SerializableFactory, SerializableBase

    ser_factory = SerializableFactory()

    class MyObject(SerializableBase):
        def __init__(self, a: int, b: str):
            super().__init()
            self.a = a
            self.b = b

        @staticmethod
        def serialize(param: 'MyObject') -> bytes:
            """ Implement the MyObject type serialization """

            bytes_string = b''
            bytes_string += ser_factory.get_apply_serializer(param.a)
            bytes_string += ser_factory.get_apply_serializer(param.b)
            return bytes_string

        @staticmethod
        def deserialize(bytes_str: bytes) -> tuple['MyObject', bytes]:
            """ Implements deserialization into a MyObject object from bytes """
            a, remaining_bytes = ser_factory.get_apply_deserializer(bytes_str, only_object=False)
            b, remaining_bytes = ser_factory.get_apply_deserializer(remaining_bytes, only_object=False)

            return MyObject(a, b), remaining_bytes


.. note::

    The ``get_apply_deserializer`` factory method has a name argument (*only_object*) being a boolean. if True, the
    returned object is the deserialized object itself otherwise it is a tuple containing the object AND the eventual
    remaining bytes