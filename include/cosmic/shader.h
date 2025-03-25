#ifndef SHADER_H
#define SHADER_H

#include <glad/glad.h>
#include <filesystem>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Shader {
public:
	// shader program ID for create, attach, link
	unsigned int ID;

	// constructor: read and build shader
	Shader(const char* vertexPath, const char* fragmentPath) {
        // Use filesystem to handle paths
        std::filesystem::path shaderDir = "../shaders";
        std::filesystem::path fullVertexPath = shaderDir / vertexPath;
        std::filesystem::path fullFragmentPath = shaderDir / fragmentPath;

		// get vertex & fragment source from filePath
		std::string vertexCode;
		std::string fragmentCode;
		std::ifstream vShaderFile;
		std::ifstream fShaderFile;
		vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		try {
			vShaderFile.open(fullVertexPath);
			std::stringstream vShaderStream;
			vShaderStream << vShaderFile.rdbuf();
			vShaderFile.close();
			vertexCode = vShaderStream.str();

			fShaderFile.open(fullFragmentPath);
			std::stringstream fShaderStream;
			fShaderStream << fShaderFile.rdbuf();
			fShaderFile.close();
			fragmentCode = fShaderStream.str();
		}
		catch (std::ifstream::failure e) {
			std::cout << "ERROR::SHADER::FILE_NOT_READ" << std::endl;
		}

		const char* vShaderCode = vertexCode.c_str();
		const char* fShaderCode = fragmentCode.c_str();

		// compile shader
		unsigned int vertex;
		unsigned int fragment;
		int success;
		char infoLog[512];

		// vShader
		vertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex, 1, &vShaderCode, NULL);
		glCompileShader(vertex);
		// error handling
		glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(vertex, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		}

		// fShader
		fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fShaderCode, NULL);
		glCompileShader(fragment);
		// error handling
		glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(fragment, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		}

		// create, attach, link shader program
		ID = glCreateProgram();
		glAttachShader(ID, vertex);
		glAttachShader(ID, fragment);
		glLinkProgram(ID);
		// errors
		glGetProgramiv(ID, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(ID, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		}

		// delete shaders after link
		glDeleteShader(vertex);
		glDeleteShader(fragment);
	}

	// use shader
	void use() {
		glUseProgram(ID);
	}
	// utilities
	void setBool(const std::string& name, bool value) const {
		glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
	}
	void setInt(const std::string& name, int value) const {
		glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
	}
	void setFloat(const std::string& name, float value) const {
		glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
	}
	void updateColor(const std::string& name) {
		float timevalue = glfwGetTime();
		float greenValue = (sin(timevalue) / 2.0f) + 0.5f;
		int vertexColorLocation = glGetUniformLocation(ID, name.c_str());
		glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);
	}
	void moveTriangle(const std::string& name) {
		// move triangle left and right via sin
		float timevalue = glfwGetTime();
		float offset = sin(timevalue) / 2.0f;
		int vertexLocation = glGetUniformLocation(ID, name.c_str());
		glUniform1f(vertexLocation, offset);
	}
	void moveTriangleC(const std::string& name) {
		// move triangle left and right via sin
		float timevalue = glfwGetTime();
		float offset = cosf(timevalue) / 2.0f;
		int vertexLocation = glGetUniformLocation(ID, name.c_str());
		glUniform1f(vertexLocation, offset);
	}
};

#endif
