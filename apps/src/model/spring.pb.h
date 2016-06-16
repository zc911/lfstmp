// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: spring.proto

#ifndef PROTOBUF_spring_2eproto__INCLUDED
#define PROTOBUF_spring_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3000000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
#include "common.pb.h"
// @@protoc_insertion_point(includes)

namespace dg {
namespace model {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_spring_2eproto();
void protobuf_AssignDesc_spring_2eproto();
void protobuf_ShutdownFile_spring_2eproto();

class VehicleObj;

// ===================================================================

class VehicleObj : public ::google::protobuf::Message {
 public:
  VehicleObj();
  virtual ~VehicleObj();

  VehicleObj(const VehicleObj& from);

  inline VehicleObj& operator=(const VehicleObj& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const VehicleObj& default_instance();

  void Swap(VehicleObj* other);

  // implements Message ----------------------------------------------

  inline VehicleObj* New() const { return New(NULL); }

  VehicleObj* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const VehicleObj& from);
  void MergeFrom(const VehicleObj& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(VehicleObj* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional .dg.model.RecVehicle Vehicle = 1;
  bool has_vehicle() const;
  void clear_vehicle();
  static const int kVehicleFieldNumber = 1;
  const ::dg::model::RecVehicle& vehicle() const;
  ::dg::model::RecVehicle* mutable_vehicle();
  ::dg::model::RecVehicle* release_vehicle();
  void set_allocated_vehicle(::dg::model::RecVehicle* vehicle);

  // optional .dg.model.StorageConfig StorageInfo = 2;
  bool has_storageinfo() const;
  void clear_storageinfo();
  static const int kStorageInfoFieldNumber = 2;
  const ::dg::model::StorageConfig& storageinfo() const;
  ::dg::model::StorageConfig* mutable_storageinfo();
  ::dg::model::StorageConfig* release_storageinfo();
  void set_allocated_storageinfo(::dg::model::StorageConfig* storageinfo);

  // optional .dg.model.Image Img = 3;
  bool has_img() const;
  void clear_img();
  static const int kImgFieldNumber = 3;
  const ::dg::model::Image& img() const;
  ::dg::model::Image* mutable_img();
  ::dg::model::Image* release_img();
  void set_allocated_img(::dg::model::Image* img);

  // @@protoc_insertion_point(class_scope:dg.model.VehicleObj)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  bool _is_default_instance_;
  ::dg::model::RecVehicle* vehicle_;
  ::dg::model::StorageConfig* storageinfo_;
  ::dg::model::Image* img_;
  mutable int _cached_size_;
  friend void  protobuf_AddDesc_spring_2eproto();
  friend void protobuf_AssignDesc_spring_2eproto();
  friend void protobuf_ShutdownFile_spring_2eproto();

  void InitAsDefaultInstance();
  static VehicleObj* default_instance_;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// VehicleObj

// optional .dg.model.RecVehicle Vehicle = 1;
inline bool VehicleObj::has_vehicle() const {
  return !_is_default_instance_ && vehicle_ != NULL;
}
inline void VehicleObj::clear_vehicle() {
  if (GetArenaNoVirtual() == NULL && vehicle_ != NULL) delete vehicle_;
  vehicle_ = NULL;
}
inline const ::dg::model::RecVehicle& VehicleObj::vehicle() const {
  // @@protoc_insertion_point(field_get:dg.model.VehicleObj.Vehicle)
  return vehicle_ != NULL ? *vehicle_ : *default_instance_->vehicle_;
}
inline ::dg::model::RecVehicle* VehicleObj::mutable_vehicle() {
  
  if (vehicle_ == NULL) {
    vehicle_ = new ::dg::model::RecVehicle;
  }
  // @@protoc_insertion_point(field_mutable:dg.model.VehicleObj.Vehicle)
  return vehicle_;
}
inline ::dg::model::RecVehicle* VehicleObj::release_vehicle() {
  
  ::dg::model::RecVehicle* temp = vehicle_;
  vehicle_ = NULL;
  return temp;
}
inline void VehicleObj::set_allocated_vehicle(::dg::model::RecVehicle* vehicle) {
  delete vehicle_;
  vehicle_ = vehicle;
  if (vehicle) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:dg.model.VehicleObj.Vehicle)
}

// optional .dg.model.StorageConfig StorageInfo = 2;
inline bool VehicleObj::has_storageinfo() const {
  return !_is_default_instance_ && storageinfo_ != NULL;
}
inline void VehicleObj::clear_storageinfo() {
  if (GetArenaNoVirtual() == NULL && storageinfo_ != NULL) delete storageinfo_;
  storageinfo_ = NULL;
}
inline const ::dg::model::StorageConfig& VehicleObj::storageinfo() const {
  // @@protoc_insertion_point(field_get:dg.model.VehicleObj.StorageInfo)
  return storageinfo_ != NULL ? *storageinfo_ : *default_instance_->storageinfo_;
}
inline ::dg::model::StorageConfig* VehicleObj::mutable_storageinfo() {
  
  if (storageinfo_ == NULL) {
    storageinfo_ = new ::dg::model::StorageConfig;
  }
  // @@protoc_insertion_point(field_mutable:dg.model.VehicleObj.StorageInfo)
  return storageinfo_;
}
inline ::dg::model::StorageConfig* VehicleObj::release_storageinfo() {
  
  ::dg::model::StorageConfig* temp = storageinfo_;
  storageinfo_ = NULL;
  return temp;
}
inline void VehicleObj::set_allocated_storageinfo(::dg::model::StorageConfig* storageinfo) {
  delete storageinfo_;
  storageinfo_ = storageinfo;
  if (storageinfo) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:dg.model.VehicleObj.StorageInfo)
}

// optional .dg.model.Image Img = 3;
inline bool VehicleObj::has_img() const {
  return !_is_default_instance_ && img_ != NULL;
}
inline void VehicleObj::clear_img() {
  if (GetArenaNoVirtual() == NULL && img_ != NULL) delete img_;
  img_ = NULL;
}
inline const ::dg::model::Image& VehicleObj::img() const {
  // @@protoc_insertion_point(field_get:dg.model.VehicleObj.Img)
  return img_ != NULL ? *img_ : *default_instance_->img_;
}
inline ::dg::model::Image* VehicleObj::mutable_img() {
  
  if (img_ == NULL) {
    img_ = new ::dg::model::Image;
  }
  // @@protoc_insertion_point(field_mutable:dg.model.VehicleObj.Img)
  return img_;
}
inline ::dg::model::Image* VehicleObj::release_img() {
  
  ::dg::model::Image* temp = img_;
  img_ = NULL;
  return temp;
}
inline void VehicleObj::set_allocated_img(::dg::model::Image* img) {
  delete img_;
  img_ = img;
  if (img) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:dg.model.VehicleObj.Img)
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace model
}  // namespace dg

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_spring_2eproto__INCLUDED
