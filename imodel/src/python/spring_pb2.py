# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: spring.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='spring.proto',
  package='dg.model',
  syntax='proto3',
  serialized_pb=_b('\n\x0cspring.proto\x12\x08\x64g.model\x1a\x0c\x63ommon.proto2\x8d\x01\n\rSpringService\x12=\n\x0cIndexVehicle\x12\x14.dg.model.VehicleObj\x1a\x15.dg.model.NullMessage\"\x00\x12=\n\x0c\x42ingoVehicle\x12\x14.dg.model.VehicleObj\x1a\x15.dg.model.NullMessage\"\x00\x62\x06proto3')
  ,
  dependencies=[common__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)





import grpc
from grpc.beta import implementations as beta_implementations
from grpc.beta import interfaces as beta_interfaces
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities


class SpringServiceStub(object):
  """## Business Intelligence APIs
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.IndexVehicle = channel.unary_unary(
        '/dg.model.SpringService/IndexVehicle',
        request_serializer=common__pb2.VehicleObj.SerializeToString,
        response_deserializer=common__pb2.NullMessage.FromString,
        )
    self.BingoVehicle = channel.unary_unary(
        '/dg.model.SpringService/BingoVehicle',
        request_serializer=common__pb2.VehicleObj.SerializeToString,
        response_deserializer=common__pb2.NullMessage.FromString,
        )


class SpringServiceServicer(object):
  """## Business Intelligence APIs
  """

  def IndexVehicle(self, request, context):
    """### Index APIs
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def BingoVehicle(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_SpringServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'IndexVehicle': grpc.unary_unary_rpc_method_handler(
          servicer.IndexVehicle,
          request_deserializer=common__pb2.VehicleObj.FromString,
          response_serializer=common__pb2.NullMessage.SerializeToString,
      ),
      'BingoVehicle': grpc.unary_unary_rpc_method_handler(
          servicer.BingoVehicle,
          request_deserializer=common__pb2.VehicleObj.FromString,
          response_serializer=common__pb2.NullMessage.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'dg.model.SpringService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class BetaSpringServiceServicer(object):
  """## Business Intelligence APIs
  """
  def IndexVehicle(self, request, context):
    """### Index APIs
    """
    context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
  def BingoVehicle(self, request, context):
    context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)


class BetaSpringServiceStub(object):
  """## Business Intelligence APIs
  """
  def IndexVehicle(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
    """### Index APIs
    """
    raise NotImplementedError()
  IndexVehicle.future = None
  def BingoVehicle(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
    raise NotImplementedError()
  BingoVehicle.future = None


def beta_create_SpringService_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
  request_deserializers = {
    ('dg.model.SpringService', 'BingoVehicle'): common__pb2.VehicleObj.FromString,
    ('dg.model.SpringService', 'IndexVehicle'): common__pb2.VehicleObj.FromString,
  }
  response_serializers = {
    ('dg.model.SpringService', 'BingoVehicle'): common__pb2.NullMessage.SerializeToString,
    ('dg.model.SpringService', 'IndexVehicle'): common__pb2.NullMessage.SerializeToString,
  }
  method_implementations = {
    ('dg.model.SpringService', 'BingoVehicle'): face_utilities.unary_unary_inline(servicer.BingoVehicle),
    ('dg.model.SpringService', 'IndexVehicle'): face_utilities.unary_unary_inline(servicer.IndexVehicle),
  }
  server_options = beta_implementations.server_options(request_deserializers=request_deserializers, response_serializers=response_serializers, thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
  return beta_implementations.server(method_implementations, options=server_options)


def beta_create_SpringService_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
  request_serializers = {
    ('dg.model.SpringService', 'BingoVehicle'): common__pb2.VehicleObj.SerializeToString,
    ('dg.model.SpringService', 'IndexVehicle'): common__pb2.VehicleObj.SerializeToString,
  }
  response_deserializers = {
    ('dg.model.SpringService', 'BingoVehicle'): common__pb2.NullMessage.FromString,
    ('dg.model.SpringService', 'IndexVehicle'): common__pb2.NullMessage.FromString,
  }
  cardinalities = {
    'BingoVehicle': cardinality.Cardinality.UNARY_UNARY,
    'IndexVehicle': cardinality.Cardinality.UNARY_UNARY,
  }
  stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, request_serializers=request_serializers, response_deserializers=response_deserializers, thread_pool=pool, thread_pool_size=pool_size)
  return beta_implementations.dynamic_stub(channel, 'dg.model.SpringService', cardinalities, options=stub_options)
# @@protoc_insertion_point(module_scope)