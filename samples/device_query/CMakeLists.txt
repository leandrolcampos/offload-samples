add_executable(
  device-query
  main.cpp
)

target_link_libraries(
  device-query PRIVATE
  OLSCommon
)