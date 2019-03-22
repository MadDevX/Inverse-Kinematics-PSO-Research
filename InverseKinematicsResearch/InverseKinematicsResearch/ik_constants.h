#pragma once
#include <glm/glm.hpp>
#define blockSize 256
#define DEGREES_OF_FREEDOM 9
#define GIZMO_SIZE 0.2f
#define GIZMO_SCALE_MATRIX glm::scale(glm::mat4(1.0f), glm::vec3(GIZMO_SIZE))
#define PI glm::pi<float>()