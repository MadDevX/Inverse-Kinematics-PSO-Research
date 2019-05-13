#include<stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <LearnOpenGL/Shader.h>
#include <im3d/im3d.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <ccd/ccd.h>
#include <ccd/quat.h>
#include "Models.h"
#include "Particle.h"
#include "Node.h"
int WINDOW_WIDTH = 800;
int WINDOW_HEIGHT = 600;
int N = 4096;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

double lastX = 0.0f, lastY = 0.0f;
float rotationY = 0.0f;
bool rotate = false;

glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -5.0f));

extern cudaError_t checkCollision(float x1, float x2);
extern cudaError_t initGenerators(curandState_t *randoms, int size);
extern cudaError_t calculatePSONew(ParticleNew *particles, float *bests, curandState_t *randoms, int size, NodeCUDA *chain, Config config, CoordinatesNew *result);
GLFWwindow* initOpenGLContext();
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow *window);
void calculateDeltaTime();
unsigned int initVAO(unsigned int *VBO);
unsigned int initCoordVAO(unsigned int *VBO);
void drawCoordinates(Shader shader, unsigned int VAO);

TargetNode* movingTarget;
OriginNode* nodeArm;
TargetNode** targets;

struct obj_t
{
	float x, y, z;
	ccd_vec3_t pos;
	ccd_quat_t quat;
};

void support(const void *_obj, const ccd_vec3_t *_dir, ccd_vec3_t *v)
{
	// assume that obj_t is user-defined structure that holds info about
	// object (in this case box: x, y, z, pos, quat - dimensions of box,
	// position and rotation)
	obj_t *obj = (obj_t *)_obj;
	ccd_vec3_t dir;
	ccd_quat_t qinv;

	// apply rotation on direction vector
	ccdVec3Copy(&dir, _dir);
	ccdQuatInvert2(&qinv, &obj->quat);
	ccdQuatRotVec(&dir, &qinv);

	// compute support point in specified direction
	ccdVec3Set(v, 
		ccdSign(ccdVec3X(&dir)) * obj->x * CCD_REAL(0.5),
		ccdSign(ccdVec3Y(&dir)) * obj->y * CCD_REAL(0.5),
		ccdSign(ccdVec3Z(&dir)) * obj->z * CCD_REAL(0.5));
	// czlowiek to kubek q.e.d.
	// transform support point according to position and rotation of object
	ccdQuatRotVec(v, &obj->quat);
	ccdVec3Add(v, &obj->pos);
}


int main(int argc, char** argv)
{
	if (argc == 2)
	{
		N = atoi(argv[1]);
	}
	obj_t box1, box2;
	ccd_vec3_t position;
	ccd_quat_t rotation;
	ccdQuatSet(&rotation, 0.0f, 0.0f, 0.0f, 1.0f);
	box2.quat = box1.quat = rotation;
	box2.x = box1.x = 1.0f;
	box2.y = box1.y = 1.0f;
	box2.z = box1.z = 1.0f;


	ccdVec3Set(&position, 5.0f, 0.0f, 0.0f);
	box1.pos = position;
	ccdVec3Set(&position, 4.000001f, 0.0f, 0.0f);
	box2.pos = position;

	ccd_t ccd;
	CCD_INIT(&ccd);

	ccd.support1 = support;
	ccd.support2 = support;
	ccd.max_iterations = 100;
	int intersect = ccdGJKIntersect(&box1, &box2, &ccd);
	printf("intersect result: %d\n", intersect);

	checkCollision(5.0f, 3.999f);

	#pragma region GLfunctions

	GLFWwindow* window = initOpenGLContext();
	Shader shader("3.3.jointShader.vert", "3.3.jointShader.frag");
	unsigned int VBO, coordVBO, linkVBO;
	unsigned int VAO = initVAO(&VBO);
	unsigned int coordVAO = initCoordVAO(&coordVBO);
	#pragma endregion

    nodeArm = new OriginNode(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(2*PI));
	int elbows = 4;
	Node** nodeElbows = new Node*[elbows];
	for (int i = 0; i < elbows; i++)
	{
		nodeElbows[i] = new Node(glm::vec3(0.0f, 1.57f, 0.0f), glm::vec3(0.0f), glm::vec3(2 * PI), 1.0f);
	}
	EffectorNode* nodeWrist = new EffectorNode(glm::vec3(0.0f, 1.57f, 0.0f), glm::vec3(0.0f), glm::vec3(2 * PI), 1.0f);
	EffectorNode* nodeWrist2 = new EffectorNode(glm::vec3(0.0f, 0.0f, 1.57f), glm::vec3(0.0f), glm::vec3(2 * PI), 1.0f);
	EffectorNode* nodeWrist3 = new EffectorNode(glm::vec3(0.0f, 0.0f, 1.57f), glm::vec3(0.0f), glm::vec3(2 * PI), 1.0f);
	TargetNode* nodeTarget1 = new TargetNode(glm::vec3(1.0f, 1.0f, -1.5f));
	TargetNode* nodeTarget2 = new TargetNode(glm::vec3(-1.0f, 1.0f, -1.5f));
	TargetNode* nodeTarget3 = new TargetNode(glm::vec3(0.0f, 0.0f, -2.0f));
	
	movingTarget = nodeTarget1;
	targets = (TargetNode**)malloc(AMOUNT_OF_TARGETS * sizeof(TargetNode*));
	
	targets[0] = nodeTarget1;
	targets[1] = nodeTarget2;
	targets[2] = nodeTarget3;

	nodeWrist->target = nodeTarget1;
	nodeWrist2->target = nodeTarget2;
	nodeWrist3->target = nodeTarget3;



	nodeArm->AttachChild(nodeElbows[0]);
	for (int i = 1; i < elbows; i++)
	{
		nodeElbows[i - 1]->AttachChild(nodeElbows[i]);
	}
	nodeElbows[elbows-1]->AttachChild(nodeWrist);
	nodeElbows[elbows-1]->AttachChild(nodeWrist2);
	nodeElbows[elbows-1]->AttachChild(nodeWrist3);
	
	NodeCUDA* chainCuda = nodeArm->AllocateCUDA();
	curandState_t *randoms;
	ParticleNew *particles;

	Config config;
	float *bests;
	
	cudaMalloc((void**)&randoms, N * sizeof(curandState_t));
	cudaMallocManaged((void**)&particles, N * sizeof(ParticleNew));
	cudaMallocManaged((void**)&bests, N * sizeof(float));
	initGenerators(randoms, N);

	while (!glfwWindowShouldClose(window))
	{
		calculateDeltaTime();
		processInput(window);
		glfwPollEvents();

		cudaError_t status;
		CoordinatesNew coords;

		nodeArm->ToCUDA(chainCuda);
		status = calculatePSONew(particles, bests, randoms, N, chainCuda, config, &coords);
		if (status != cudaSuccess) break;
		int ind = 0;
		nodeArm->FromCoords(coords,&ind);
		
		#pragma region GLrendering

		glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		shader.use();
		shader.setMat4("view", view);
		shader.setMat4("projection", glm::perspective(glm::radians(45.0f), (float)WINDOW_WIDTH / WINDOW_HEIGHT, 0.1f, 100.0f));
		drawCoordinates(shader, coordVAO);
		nodeArm->Draw(shader, VAO);
		nodeTarget1->DrawCurrent(shader,VAO);
		nodeTarget2->DrawCurrent(shader, VAO);
		nodeTarget3->DrawCurrent(shader, VAO);

		glfwSwapBuffers(window);
		
		#pragma endregion

	}
	
	#pragma region cleanup
	glDeleteVertexArrays(1, &VAO);
	glDeleteVertexArrays(1, &coordVAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &coordVBO);

	glfwTerminate();

	cudaFree(bests);
	cudaFree(chainCuda);
	cudaFree(particles);
	cudaFree(randoms);
	delete(nodeArm);
	
	for (int i = 0; i < elbows; i++)
	{
		delete(nodeElbows[i]);
	}
	delete[](nodeElbows);
	delete(nodeWrist);
	delete(nodeWrist2);
	delete(nodeWrist3);
	#pragma endregion

	
	return 0;
}

unsigned int initVAO(unsigned int *VBO)
{
	unsigned int VAO;
	glGenVertexArrays(1, &VAO);
	
	glGenBuffers(1, VBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, *VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VERTICES_CUBE), VERTICES_CUBE, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	return VAO;
}

unsigned int initCoordVAO(unsigned int *VBO)
{
	unsigned int VAO;
	glGenVertexArrays(1, &VAO);

	glGenBuffers(1, VBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, *VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VERTICES_GRID), VERTICES_GRID, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	return VAO;
}



void drawCoordinates(Shader shader, unsigned int VAO)
{
	shader.use();
	glBindVertexArray(VAO);
	glLineWidth(2.0f);
	shader.setMat4("view", view);
	shader.setMat4("model", glm::mat4(1.0f));
	shader.setVec3("color", 1.0f, 0.0f, 0.0f);
	glDrawArrays(GL_LINES, 0, 2);
	shader.setVec3("color", 0.0f, 1.0f, 0.0f);
	glDrawArrays(GL_LINES, 2, 2);
	shader.setVec3("color", 0.0f, 0.0f, 1.0f);
	glDrawArrays(GL_LINES, 4, 2);
}


void configureGLFWContext()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}

void processInput(GLFWwindow *window)
{

	for (int i = 0; i < AMOUNT_OF_TARGETS; i++)
	{
		if (glfwGetKey(window, GLFW_KEY_1+i) == GLFW_PRESS)
			movingTarget = (targets[i]);
	}

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
		nodeArm->translate(glm::vec3(-1.0f, 0.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
		nodeArm->translate(glm::vec3(1.0f, 0.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
		nodeArm->translate(glm::vec3(0.0f, 1.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
		nodeArm->translate(glm::vec3(0.0f, -1.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		movingTarget->translate(glm::vec3(-1.0f, 0.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		movingTarget->translate(glm::vec3(1.0f, 0.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		movingTarget->translate(glm::vec3(0.0f, 1.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		movingTarget->translate(glm::vec3(0.0f, -1.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		movingTarget->translate(glm::vec3(0.0f, 0.0f, 1.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		movingTarget->translate(glm::vec3(0.0f, 0.0f, -1.0f) * deltaTime);
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
	{
		rotate = true;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	else
	{
		rotate = false;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

GLFWwindow* createWindow()
{
	GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Inverse Kinematics PSO", NULL, NULL);
	if (window == NULL)
	{
		glfwTerminate();
		throw - 1;
	}
	glfwMakeContextCurrent(window);

	return window;
}

GLFWwindow* initOpenGLContext()
{
	glfwInit();
	configureGLFWContext();

	GLFWwindow* window;

	try
	{
		window = createWindow();
	}
	catch (int)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		exit(EXIT_FAILURE);
	}

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	glEnable(GL_DEPTH_TEST);
	glfwSetCursorPosCallback(window, mouse_callback);

	return window;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	height = height == 0 ? 1: height;
	width = width == 0 ? 1 : width;
	glViewport(0, 0, width, height);
	WINDOW_WIDTH = width;
	WINDOW_HEIGHT = height;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{

	float xoffset = (float)xpos - lastX;
	float yoffset = lastY - (float)ypos;

	lastX = (float)xpos;
	lastY = (float)ypos;

	if(rotate)
	{
		rotationY += xoffset * 0.01f;
		glm::mat4 rotMat = glm::rotate(glm::mat4(1.0f), xoffset * 0.01f, glm::vec3(0.0f, 1.0f, 0.0f));
		view = view * rotMat;
		rotMat = glm::rotate(glm::mat4(1.0f), -yoffset * 0.01f, glm::vec3(cosf(rotationY), 0.0f, sinf(rotationY)));
		view = view * rotMat;

	}
	
}

void calculateDeltaTime()
{
	float currentFrame = (float)glfwGetTime();
	deltaTime = currentFrame - lastFrame;
	lastFrame = currentFrame;
}