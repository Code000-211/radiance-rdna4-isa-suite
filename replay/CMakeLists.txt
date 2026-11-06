# Included from Radiance tools/CMakeLists after rad_add_tool is defined.
rad_add_tool(rad-kbench-replay "${CMAKE_CURRENT_LIST_DIR}/worker.cpp")
target_include_directories(rad-kbench-replay PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
# Snapshot export is opt-in at runtime. Only tools are modified; no serving-engine hooks.
target_compile_definitions(rad-kbench PRIVATE RAD_KBENCH_REPLAY_CAPTURE=1)
target_include_directories(rad-kbench PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
rad_add_tool(rad-replay-inventory "${CMAKE_CURRENT_LIST_DIR}/inventory.cpp")
target_include_directories(rad-replay-inventory PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
