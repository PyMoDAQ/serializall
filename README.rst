SERIALIZALL
###########



Serializall is meant to serialize any python objects. It incorporates python built-ins serialization. 

It is based on a Factory pattern allowing to register objects that can be serialized. To do so objects should inherit a MixIn base class exposing a mandatory interface to two methods namely *serialize* and *deserialize*

It is actually used in projects such as PyMoDAQ and pyqtgraph.