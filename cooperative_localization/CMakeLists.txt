cmake_minimum_required(VERSION 3.8)
project(cooperative_localization)

# Configurazione di base
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Trova tutte le dipendenze
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)

# Configurazione esplicita dei percorsi di output
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Crea la libreria
add_library(logic_lib
  src/data_loader.cpp
  src/ekf_centralized.cpp
)

# Configura gli include
target_include_directories(logic_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
)

# Collega le dipendenze
ament_target_dependencies(logic_lib
  PUBLIC
    rclcpp
)
target_link_libraries(logic_lib PUBLIC Eigen3::Eigen)

# Crea l'eseguibile (VERSIONE CORRETTA)
add_executable(centralized_node 
  src/centralized_node.cpp
)
target_include_directories(centralized_node PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
ament_target_dependencies(centralized_node
  rclcpp
)
target_link_libraries(centralized_node
  logic_lib
)

# Installazione (VERSIONE FUNZIONANTE)
install(TARGETS centralized_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS logic_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()