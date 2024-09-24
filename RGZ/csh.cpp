#include <GL/glew.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

GLuint genTransformProg();
GLuint genInitProg();
void initBuffers(GLuint*& bufferID);
void transformBuffers(GLuint* bufferID);
const float alpha = 2.0f;
const int N = 1 << 10;

float* startData = new float[N];
float* result = new float[N];


void checkErrors(std::string desc) {
	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		fprintf(stderr, "OpenGL error in \"%s\": %s (%d)\n", desc.c_str(),
			gluErrorString(e), e);
		exit(20);
	}
}


void initBuffers(GLuint*& bufferID) {
	glGenBuffers(2, bufferID);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID[0]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, N * sizeof(float), 0,
		GL_DYNAMIC_DRAW);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID[1]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, N * sizeof(float), 0,
		GL_DYNAMIC_DRAW);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID[0]);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferID[1]);
	GLuint csInitID = genInitProg();
	glUseProgram(csInitID);
	glDispatchCompute(N/128 , 1 , 1);
	//glDispatchCompute(16 , 1 , 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
		GL_BUFFER_UPDATE_BARRIER_BIT);
	glDeleteProgram(csInitID);
}

GLuint genInitProg() {
	GLuint progHandle = glCreateProgram();
	GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
	const char* cpSrc[] = {
	"#version 430\n",
	"layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in; \
	layout(std430, binding = 0) buffer BufferA{float A[];};\
	layout(std430, binding = 1) buffer BufferB{float B[];};\
	void main() {\
		uint idx = gl_GlobalInvocationID.x;\
		A[idx]=float(idx);\
		B[idx]=float(idx);\
	}"
	};
	int rvalue;
	glShaderSource(cs, 2, cpSrc, NULL);
	glCompileShader(cs);
	glGetShaderiv(cs, GL_COMPILE_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in compiling fp\n");
		exit(31);
	}

	glAttachShader(progHandle, cs);
	glLinkProgram(progHandle);
	glGetProgramiv(progHandle, GL_LINK_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in linking sp\n");
		exit(32);
	}
	checkErrors("Render shaders");
	return progHandle;
}

GLuint genTransformProg() {
	GLuint progHandle = glCreateProgram();
	GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
	const char* cpSrc[] = {
	"#version 430\n",
	"layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in; \
	layout(std430, binding = 0) buffer BufferA{float A[];};\
	layout(std430, binding = 1) buffer BufferB{float B[];};\
	uniform float alpha = 2.0f;\
	void main() {\
		uint idx = gl_GlobalInvocationID.x;\
		B[idx] = alpha * A[idx] + B[idx];\
	}"

	};
	glShaderSource(cs, 2, cpSrc, NULL);
	int rvalue;
	glShaderSource(cs, 2, cpSrc, NULL);
	glCompileShader(cs);
	glGetShaderiv(cs, GL_COMPILE_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in compiling fp\n");
		exit(31);
	}

	glAttachShader(progHandle, cs);
	glLinkProgram(progHandle);
	glGetProgramiv(progHandle, GL_LINK_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in linking sp\n");
		exit(32);
	}
	checkErrors("Render shaders");
	return progHandle;
}

void transformBuffers(GLuint * bufferID) {
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID[0]);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferID[1]);
	GLuint csTransformID = genTransformProg();
	GLuint koef = glGetUniformLocation(csTransformID, "alpha");
	glUniform1f(koef, alpha);
	glUseProgram(csTransformID);
	
	//const auto start = std::chrono::high_resolution_clock::now();
	GLuint timeElapsed = 0;
	GLuint queries[1];
	glGenQueries(1, queries);
	glBeginQuery(GL_TIME_ELAPSED, queries[0]);
	//glDispatchCompute(16, 1, 1);
	std::chrono::steady_clock::time_point pr_StartTime;
	std::chrono::steady_clock::time_point pr_EndTime;
	pr_StartTime = std::chrono::steady_clock::now();
	glDispatchCompute(N/256, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
		GL_BUFFER_UPDATE_BARRIER_BIT);
	pr_EndTime = std::chrono::steady_clock::now();
	glEndQuery(GL_TIME_ELAPSED);
	
	auto Duration = std::chrono::duration_cast<std::chrono::microseconds>(pr_EndTime - pr_StartTime);
	std::cout << "Time: " << Duration.count() << " mks." << std::endl;
	glGetQueryObjectuiv(queries[0], GL_QUERY_RESULT, &timeElapsed);
    //std::cout << "Took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";
    printf("Time: %d nanoseconds, %f milliseconds\n", timeElapsed, float(timeElapsed)/1000000);
	glDeleteProgram(csTransformID);
}


void outputBuffers(GLuint* bufferID, int id) {
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID[id]);
	float* data = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER,
		GL_READ_ONLY);
	float* hdata = (float*)calloc(N, sizeof(float));
	memcpy(&hdata[0], data, sizeof(float) * N);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	if  (id == 0) {
		//printf("Data:	");
		memcpy(&startData[0], hdata, sizeof(float) * N);
	} else if (id == 1) {
		//printf("Result:	");
		memcpy(&result[0], hdata, sizeof(float) * N);
	}

	// for (int i = 0; i < N; i++) {
	// 	printf("%g\t", hdata[i]);
	// }
	// printf("\n");
	free(hdata);
}

void checkResult() {
	printf("N = %d\n", N);
	for(int i = 0; i < N; i++) {
		if(startData[i] * alpha + startData[i] == result[i]) {
			continue;
		}
		else {
			printf("Wrong answer!\n");
			return;
		}
	}
	printf("Correct answer!\n");
	return;
}
