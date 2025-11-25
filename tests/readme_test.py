from serializall.factory import SerializableFactory, SerializableBase

ser_factory = SerializableFactory()


@SerializableFactory.register_decorator()
class MyObject(SerializableBase):
    def __init__(self, a: int, b: str):
        super().__init__()
        self.a = a
        self.b = b

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b

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
        a, remaining_bytes = ser_factory.get_apply_deserializer(bytes_str, False)
        b, remaining_bytes = ser_factory.get_apply_deserializer(remaining_bytes, False)

        return MyObject(a, b), remaining_bytes


def test_readme():
    an_object_to_serialize = MyObject(124, 'arandomstring')

    serialized_object = ser_factory.get_apply_serializer(an_object_to_serialize)

    assert isinstance(ser_factory.get_apply_serializer(serialized_object), bytes)

    assert ser_factory.get_apply_deserializer(serialized_object) == an_object_to_serialize