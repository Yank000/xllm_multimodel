include(pybind_extension)

pybind_extension(
  NAME
    xllm_export
  COPTS
    -DPY_MODULE_NAME=xllm_export
  SRCS
    bind.cpp
  DEFINES
    PYBIND11_DETAILED_ERROR_MESSAGES=1
  LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
  DEPS
    :master
    :request
    :util
    absl::strings
    brpc
    gflags::gflags
    glog::glog
    Python::Module
    torch_python
    torch
    c10
)
target_link_options(xllm_export PRIVATE -Wl,-Bsymbolic)
target_link_libraries(xllm_export PRIVATE
  xtensor
  request
  prefix_cache
  block
)
target_link_options(xllm_export PRIVATE
  "-Wl,-u,_ZN4xllm23RequestMetricAggregator8instanceEv"
  "-Wl,-u,_ZNK4xllm24GlobalPrefixCacheManager23get_total_cached_blocksEv"
)
target_link_libraries(common PRIVATE leveldb::leveldb OpenSSL::SSL OpenSSL::Crypto protobuf::libprotobuf)
add_dependencies(common brpc-static)
