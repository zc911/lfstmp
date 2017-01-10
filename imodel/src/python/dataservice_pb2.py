# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: dataservice.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import deepdatasingle_pb2 as deepdatasingle__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='dataservice.proto',
  package='model',
  syntax='proto3',
  serialized_pb=_b('\n\x11\x64\x61taservice.proto\x12\x05model\x1a\x14\x64\x65\x65pdatasingle.proto\"0\n\x0b\x44\x61taRequest\x12!\n\x06\x45ntity\x18\x01 \x01(\x0b\x32\x11.model.GenericObj\"7\n\x10\x42\x61tchDataRequest\x12#\n\x08\x45ntities\x18\x02 \x03(\x0b\x32\x11.model.GenericObj\"K\n\x0c\x44\x61taResponse\x12 \n\x06Status\x18\x01 \x01(\x0e\x32\x10.model.ApiStatus\x12\x0b\n\x03Msg\x18\x02 \x01(\t\x12\x0c\n\x04\x44\x61ta\x18\x03 \x01(\x0c\"3\n\rSearchRequest\x12\"\n\x05Query\x18\x01 \x01(\x0b\x32\x13.model.GenericQuery\"s\n\x0cGenericQuery\x12\x1c\n\x04Type\x18\x01 \x01(\x0e\x32\x0e.model.ObjType\x12#\n\x07\x46mtType\x18\x02 \x01(\x0e\x32\x12.model.DataFmtType\x12\x0f\n\x07StrData\x18\x03 \x01(\t\x12\x0f\n\x07\x42inData\x18\x04 \x01(\x0c\"q\n\nGenericObj\x12\x1c\n\x04Type\x18\x01 \x01(\x0e\x32\x0e.model.ObjType\x12#\n\x07\x46mtType\x18\x02 \x01(\x0e\x32\x12.model.DataFmtType\x12\x0f\n\x07StrData\x18\x03 \x01(\t\x12\x0f\n\x07\x42inData\x18\x04 \x01(\x0c\"G\n\x08HttpResp\x12 \n\x06Status\x18\x01 \x01(\x0e\x32\x10.model.ApiStatus\x12\x0b\n\x03Msg\x18\x02 \x01(\t\x12\x0c\n\x04\x44\x61ta\x18\x03 \x01(\t\".\n\x0bGenericObjs\x12\x1f\n\x04Objs\x18\x01 \x03(\x0b\x32\x11.model.GenericObj\"I\n\nInsertResp\x12 \n\x06Status\x18\x01 \x01(\x0e\x32\x10.model.ApiStatus\x12\x0b\n\x03Msg\x18\x02 \x01(\t\x12\x0c\n\x04\x44\x61ta\x18\x03 \x01(\t*6\n\tApiStatus\x12\x11\n\rUNKNOWNSTATUS\x10\x00\x12\x0b\n\x07SUCCESS\x10\x01\x12\t\n\x05\x45RROR\x10\x02\x32\xbc\x01\n\x0b\x44\x61taService\x12\x35\n\x08SendData\x12\x12.model.DataRequest\x1a\x13.model.DataResponse\"\x00\x12?\n\rSendBatchData\x12\x17.model.BatchDataRequest\x1a\x13.model.DataResponse\"\x00\x12\x35\n\x06Search\x12\x14.model.SearchRequest\x1a\x13.model.DataResponse\"\x00\x62\x06proto3')
  ,
  dependencies=[deepdatasingle__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

_APISTATUS = _descriptor.EnumDescriptor(
  name='ApiStatus',
  full_name='model.ApiStatus',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWNSTATUS', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SUCCESS', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=715,
  serialized_end=769,
)
_sym_db.RegisterEnumDescriptor(_APISTATUS)

ApiStatus = enum_type_wrapper.EnumTypeWrapper(_APISTATUS)
UNKNOWNSTATUS = 0
SUCCESS = 1
ERROR = 2



_DATAREQUEST = _descriptor.Descriptor(
  name='DataRequest',
  full_name='model.DataRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Entity', full_name='model.DataRequest.Entity', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=50,
  serialized_end=98,
)


_BATCHDATAREQUEST = _descriptor.Descriptor(
  name='BatchDataRequest',
  full_name='model.BatchDataRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Entities', full_name='model.BatchDataRequest.Entities', index=0,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=100,
  serialized_end=155,
)


_DATARESPONSE = _descriptor.Descriptor(
  name='DataResponse',
  full_name='model.DataResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Status', full_name='model.DataResponse.Status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='Msg', full_name='model.DataResponse.Msg', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='Data', full_name='model.DataResponse.Data', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=157,
  serialized_end=232,
)


_SEARCHREQUEST = _descriptor.Descriptor(
  name='SearchRequest',
  full_name='model.SearchRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Query', full_name='model.SearchRequest.Query', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=234,
  serialized_end=285,
)


_GENERICQUERY = _descriptor.Descriptor(
  name='GenericQuery',
  full_name='model.GenericQuery',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Type', full_name='model.GenericQuery.Type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='FmtType', full_name='model.GenericQuery.FmtType', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='StrData', full_name='model.GenericQuery.StrData', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='BinData', full_name='model.GenericQuery.BinData', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=287,
  serialized_end=402,
)


_GENERICOBJ = _descriptor.Descriptor(
  name='GenericObj',
  full_name='model.GenericObj',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Type', full_name='model.GenericObj.Type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='FmtType', full_name='model.GenericObj.FmtType', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='StrData', full_name='model.GenericObj.StrData', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='BinData', full_name='model.GenericObj.BinData', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=404,
  serialized_end=517,
)


_HTTPRESP = _descriptor.Descriptor(
  name='HttpResp',
  full_name='model.HttpResp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Status', full_name='model.HttpResp.Status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='Msg', full_name='model.HttpResp.Msg', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='Data', full_name='model.HttpResp.Data', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=519,
  serialized_end=590,
)


_GENERICOBJS = _descriptor.Descriptor(
  name='GenericObjs',
  full_name='model.GenericObjs',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Objs', full_name='model.GenericObjs.Objs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=592,
  serialized_end=638,
)


_INSERTRESP = _descriptor.Descriptor(
  name='InsertResp',
  full_name='model.InsertResp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Status', full_name='model.InsertResp.Status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='Msg', full_name='model.InsertResp.Msg', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='Data', full_name='model.InsertResp.Data', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=640,
  serialized_end=713,
)

_DATAREQUEST.fields_by_name['Entity'].message_type = _GENERICOBJ
_BATCHDATAREQUEST.fields_by_name['Entities'].message_type = _GENERICOBJ
_DATARESPONSE.fields_by_name['Status'].enum_type = _APISTATUS
_SEARCHREQUEST.fields_by_name['Query'].message_type = _GENERICQUERY
_GENERICQUERY.fields_by_name['Type'].enum_type = deepdatasingle__pb2._OBJTYPE
_GENERICQUERY.fields_by_name['FmtType'].enum_type = deepdatasingle__pb2._DATAFMTTYPE
_GENERICOBJ.fields_by_name['Type'].enum_type = deepdatasingle__pb2._OBJTYPE
_GENERICOBJ.fields_by_name['FmtType'].enum_type = deepdatasingle__pb2._DATAFMTTYPE
_HTTPRESP.fields_by_name['Status'].enum_type = _APISTATUS
_GENERICOBJS.fields_by_name['Objs'].message_type = _GENERICOBJ
_INSERTRESP.fields_by_name['Status'].enum_type = _APISTATUS
DESCRIPTOR.message_types_by_name['DataRequest'] = _DATAREQUEST
DESCRIPTOR.message_types_by_name['BatchDataRequest'] = _BATCHDATAREQUEST
DESCRIPTOR.message_types_by_name['DataResponse'] = _DATARESPONSE
DESCRIPTOR.message_types_by_name['SearchRequest'] = _SEARCHREQUEST
DESCRIPTOR.message_types_by_name['GenericQuery'] = _GENERICQUERY
DESCRIPTOR.message_types_by_name['GenericObj'] = _GENERICOBJ
DESCRIPTOR.message_types_by_name['HttpResp'] = _HTTPRESP
DESCRIPTOR.message_types_by_name['GenericObjs'] = _GENERICOBJS
DESCRIPTOR.message_types_by_name['InsertResp'] = _INSERTRESP
DESCRIPTOR.enum_types_by_name['ApiStatus'] = _APISTATUS

DataRequest = _reflection.GeneratedProtocolMessageType('DataRequest', (_message.Message,), dict(
  DESCRIPTOR = _DATAREQUEST,
  __module__ = 'dataservice_pb2'
  # @@protoc_insertion_point(class_scope:model.DataRequest)
  ))
_sym_db.RegisterMessage(DataRequest)

BatchDataRequest = _reflection.GeneratedProtocolMessageType('BatchDataRequest', (_message.Message,), dict(
  DESCRIPTOR = _BATCHDATAREQUEST,
  __module__ = 'dataservice_pb2'
  # @@protoc_insertion_point(class_scope:model.BatchDataRequest)
  ))
_sym_db.RegisterMessage(BatchDataRequest)

DataResponse = _reflection.GeneratedProtocolMessageType('DataResponse', (_message.Message,), dict(
  DESCRIPTOR = _DATARESPONSE,
  __module__ = 'dataservice_pb2'
  # @@protoc_insertion_point(class_scope:model.DataResponse)
  ))
_sym_db.RegisterMessage(DataResponse)

SearchRequest = _reflection.GeneratedProtocolMessageType('SearchRequest', (_message.Message,), dict(
  DESCRIPTOR = _SEARCHREQUEST,
  __module__ = 'dataservice_pb2'
  # @@protoc_insertion_point(class_scope:model.SearchRequest)
  ))
_sym_db.RegisterMessage(SearchRequest)

GenericQuery = _reflection.GeneratedProtocolMessageType('GenericQuery', (_message.Message,), dict(
  DESCRIPTOR = _GENERICQUERY,
  __module__ = 'dataservice_pb2'
  # @@protoc_insertion_point(class_scope:model.GenericQuery)
  ))
_sym_db.RegisterMessage(GenericQuery)

GenericObj = _reflection.GeneratedProtocolMessageType('GenericObj', (_message.Message,), dict(
  DESCRIPTOR = _GENERICOBJ,
  __module__ = 'dataservice_pb2'
  # @@protoc_insertion_point(class_scope:model.GenericObj)
  ))
_sym_db.RegisterMessage(GenericObj)

HttpResp = _reflection.GeneratedProtocolMessageType('HttpResp', (_message.Message,), dict(
  DESCRIPTOR = _HTTPRESP,
  __module__ = 'dataservice_pb2'
  # @@protoc_insertion_point(class_scope:model.HttpResp)
  ))
_sym_db.RegisterMessage(HttpResp)

GenericObjs = _reflection.GeneratedProtocolMessageType('GenericObjs', (_message.Message,), dict(
  DESCRIPTOR = _GENERICOBJS,
  __module__ = 'dataservice_pb2'
  # @@protoc_insertion_point(class_scope:model.GenericObjs)
  ))
_sym_db.RegisterMessage(GenericObjs)

InsertResp = _reflection.GeneratedProtocolMessageType('InsertResp', (_message.Message,), dict(
  DESCRIPTOR = _INSERTRESP,
  __module__ = 'dataservice_pb2'
  # @@protoc_insertion_point(class_scope:model.InsertResp)
  ))
_sym_db.RegisterMessage(InsertResp)


import grpc
from grpc.beta import implementations as beta_implementations
from grpc.beta import interfaces as beta_interfaces
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities


class DataServiceStub(object):

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.SendData = channel.unary_unary(
        '/model.DataService/SendData',
        request_serializer=DataRequest.SerializeToString,
        response_deserializer=DataResponse.FromString,
        )
    self.SendBatchData = channel.unary_unary(
        '/model.DataService/SendBatchData',
        request_serializer=BatchDataRequest.SerializeToString,
        response_deserializer=DataResponse.FromString,
        )
    self.Search = channel.unary_unary(
        '/model.DataService/Search',
        request_serializer=SearchRequest.SerializeToString,
        response_deserializer=DataResponse.FromString,
        )


class DataServiceServicer(object):

  def SendData(self, request, context):
    """SendData sends the data of a single entity to store.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SendBatchData(self, request, context):
    """SendBatchData sends the data of multiple entities to store.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Search(self, request, context):
    """Query queries stored data based on requested criteria.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_DataServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'SendData': grpc.unary_unary_rpc_method_handler(
          servicer.SendData,
          request_deserializer=DataRequest.FromString,
          response_serializer=DataResponse.SerializeToString,
      ),
      'SendBatchData': grpc.unary_unary_rpc_method_handler(
          servicer.SendBatchData,
          request_deserializer=BatchDataRequest.FromString,
          response_serializer=DataResponse.SerializeToString,
      ),
      'Search': grpc.unary_unary_rpc_method_handler(
          servicer.Search,
          request_deserializer=SearchRequest.FromString,
          response_serializer=DataResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'model.DataService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class BetaDataServiceServicer(object):
  def SendData(self, request, context):
    """SendData sends the data of a single entity to store.
    """
    context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
  def SendBatchData(self, request, context):
    """SendBatchData sends the data of multiple entities to store.
    """
    context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
  def Search(self, request, context):
    """Query queries stored data based on requested criteria.
    """
    context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)


class BetaDataServiceStub(object):
  def SendData(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
    """SendData sends the data of a single entity to store.
    """
    raise NotImplementedError()
  SendData.future = None
  def SendBatchData(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
    """SendBatchData sends the data of multiple entities to store.
    """
    raise NotImplementedError()
  SendBatchData.future = None
  def Search(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
    """Query queries stored data based on requested criteria.
    """
    raise NotImplementedError()
  Search.future = None


def beta_create_DataService_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
  request_deserializers = {
    ('model.DataService', 'Search'): SearchRequest.FromString,
    ('model.DataService', 'SendBatchData'): BatchDataRequest.FromString,
    ('model.DataService', 'SendData'): DataRequest.FromString,
  }
  response_serializers = {
    ('model.DataService', 'Search'): DataResponse.SerializeToString,
    ('model.DataService', 'SendBatchData'): DataResponse.SerializeToString,
    ('model.DataService', 'SendData'): DataResponse.SerializeToString,
  }
  method_implementations = {
    ('model.DataService', 'Search'): face_utilities.unary_unary_inline(servicer.Search),
    ('model.DataService', 'SendBatchData'): face_utilities.unary_unary_inline(servicer.SendBatchData),
    ('model.DataService', 'SendData'): face_utilities.unary_unary_inline(servicer.SendData),
  }
  server_options = beta_implementations.server_options(request_deserializers=request_deserializers, response_serializers=response_serializers, thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
  return beta_implementations.server(method_implementations, options=server_options)


def beta_create_DataService_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
  request_serializers = {
    ('model.DataService', 'Search'): SearchRequest.SerializeToString,
    ('model.DataService', 'SendBatchData'): BatchDataRequest.SerializeToString,
    ('model.DataService', 'SendData'): DataRequest.SerializeToString,
  }
  response_deserializers = {
    ('model.DataService', 'Search'): DataResponse.FromString,
    ('model.DataService', 'SendBatchData'): DataResponse.FromString,
    ('model.DataService', 'SendData'): DataResponse.FromString,
  }
  cardinalities = {
    'Search': cardinality.Cardinality.UNARY_UNARY,
    'SendBatchData': cardinality.Cardinality.UNARY_UNARY,
    'SendData': cardinality.Cardinality.UNARY_UNARY,
  }
  stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, request_serializers=request_serializers, response_deserializers=response_deserializers, thread_pool=pool, thread_pool_size=pool_size)
  return beta_implementations.dynamic_stub(channel, 'model.DataService', cardinalities, options=stub_options)
# @@protoc_insertion_point(module_scope)