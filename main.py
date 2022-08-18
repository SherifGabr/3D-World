import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyrr
import ctypes
import random
import math
import time
from ObjLoader import ObjLoader
############################## Constants ######################################

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

CONTINUE = 0
NEW_SCENE = 1
EXIT = 3



############################## helper functions ###############################

def initialise_pygame():
    pg.init()
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                pg.GL_CONTEXT_PROFILE_CORE)
    pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pg.OPENGL|pg.DOUBLEBUF)
    pg.mouse.set_pos((SCREEN_HEIGHT / 2, SCREEN_HEIGHT / 2))

def initialise_opengl():
    glEnable(GL_PROGRAM_POINT_SIZE)
    glClearColor(0.1, 0.1, 0.1, 1)
    shader3DTextured = createShader("shaders/vertex_3d_textured.txt",
                                        "shaders/fragment_3d_textured.txt")
    shader3DColored = createShader("shaders/vertex_3d_colored.txt",
                                            "shaders/fragment_3d_colored.txt")
    shader2DTextured = createShader("shaders/vertex_2d_textured.txt",
                                        "shaders/fragment_2d_textured.txt")
    shader2DColored = createShader("shaders/vertex_2d_colored.txt",
                                            "shaders/fragment_2d_colored.txt")
    shader2DText = createShader("shaders/vertex_2d_textured.txt",
                                            "shaders/fragment_text.txt")
    shader2DParticle = createShader("shaders/particle_2d_vertex.txt",
                                            "shaders/particle_2d_fragment.txt")
    shader3DBillboard = createShader("shaders/vertex_3d_billboard.txt",
                                        "shaders/fragment_3d_billboard.txt")
    shader3DCubemap = createShader("shaders/vertex_3d_cubemap.txt",
                                        "shaders/fragment_3d_cubemap.txt")
    shader3DCloth = createShader("shaders/vertex_3d_cloth.txt",
                                        "shaders/fragment_3d_cloth.txt")
                                

    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    projection_transform = pyrr.matrix44.create_perspective_projection(90, SCREEN_WIDTH/SCREEN_HEIGHT, 
                                                                        0.1, 10, dtype=np.float32)
    
    glUseProgram(shader3DTextured)
    glUniformMatrix4fv(glGetUniformLocation(shader3DTextured,"projection"),1,GL_FALSE,projection_transform)
    glUniform3fv(glGetUniformLocation(shader3DTextured,"ambient"), 1, np.array([0.1, 0.1, 0.1],dtype=np.float32))
    glUniform1i(glGetUniformLocation(shader3DTextured, "material.diffuse"), 0)
    glUniform1i(glGetUniformLocation(shader3DTextured, "material.specular"), 1)

    glUseProgram(shader3DColored)
    glUniformMatrix4fv(glGetUniformLocation(shader3DColored,"projection"),1,GL_FALSE,projection_transform)

    glUseProgram(shader2DTextured)
    glUniform1i(glGetUniformLocation(shader2DTextured, "regular_texture"), 0)
    glUniform1i(glGetUniformLocation(shader2DTextured, "bright_texture"), 1)

    glUseProgram(shader3DBillboard)
    glUniformMatrix4fv(glGetUniformLocation(shader3DBillboard,"projection"),1,GL_FALSE,projection_transform)
    glUniform1i(glGetUniformLocation(shader3DBillboard, "diffuse"), 0)

    glUseProgram(shader3DCubemap)
    glUniformMatrix4fv(glGetUniformLocation(shader3DCubemap,"projection"),1,GL_FALSE,projection_transform)
    glUniform1i(glGetUniformLocation(shader3DCubemap, "skyBox"), 0)

    glUseProgram(shader3DCloth)
    glUniformMatrix4fv(glGetUniformLocation(shader3DCloth,"projection"),1,GL_FALSE,projection_transform)
    glUniform3fv(glGetUniformLocation(shader3DCloth,"ambient"), 1, np.array([1, 1, 1],dtype=np.float32))
    glUniform1i(glGetUniformLocation(shader3DCloth, "material.diffuse"), 0)
    glUniform1i(glGetUniformLocation(shader3DCloth, "material.specular"), 1)

    return (shader3DTextured, shader3DColored, shader2DTextured, shader2DColored, shader2DText, shader2DParticle, shader3DBillboard, shader3DCubemap, shader3DCloth)

def createShader(vertexFilepath, fragmentFilepath):

        with open(vertexFilepath,'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath,'r') as f:
            fragment_src = f.readlines()
        
        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))
        
        return shader

def create_framebuffer():
    samples = 2
    multisampleFBO = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, multisampleFBO)
        
    regularCBMultisampled = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, regularCBMultisampled)
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, GL_RGB, SCREEN_WIDTH, SCREEN_HEIGHT, GL_TRUE)
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, regularCBMultisampled, 0)

    brightCBMultisampled = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, brightCBMultisampled)
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, GL_RGB, SCREEN_WIDTH, SCREEN_HEIGHT, GL_TRUE)
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D_MULTISAMPLE, brightCBMultisampled, 0)
        
    MultisampleDepthStencilBuffer = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, MultisampleDepthStencilBuffer)
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH24_STENCIL8, SCREEN_WIDTH, SCREEN_HEIGHT)
    glBindRenderbuffer(GL_RENDERBUFFER,0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, MultisampleDepthStencilBuffer)

    singlesampleFBO = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, singlesampleFBO)
        
    regularCB = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, regularCB)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                                GL_TEXTURE_2D, regularCB, 0)
    
    brightCB = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, brightCB)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 
                                GL_TEXTURE_2D, brightCB, 0)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return (
        multisampleFBO, regularCBMultisampled, brightCBMultisampled,
        MultisampleDepthStencilBuffer, singlesampleFBO, regularCB, brightCB
        )

### Models
class Cube:
    def __init__(self, shader, material, position):
        self.material = material
        self.shader = shader
        self.position = position
        glUseProgram(shader)
        # x, y, z, s, t, nx, ny, nz
        self.vertices = (
                -0.5, -0.5, -0.5, 0, 0, 0, 0, -1,
                -0.5,  0.5, -0.5, 1, 0, 0, 0, -1,
                 0.5,  0.5, -0.5, 1, 1, 0, 0, -1,

                 0.5,  0.5, -0.5, 1, 1, 0, 0, -1,
                 0.5, -0.5, -0.5, 0, 1, 0, 0, -1,
                -0.5, -0.5, -0.5, 0, 0, 0, 0, -1,

                 0.5,  0.5,  0.5, 0, 0, 0, 0,  1,
                -0.5,  0.5,  0.5, 1, 0, 0, 0,  1,
                -0.5, -0.5,  0.5, 1, 1, 0, 0,  1,

                -0.5, -0.5,  0.5, 1, 1, 0, 0,  1,
                 0.5, -0.5,  0.5, 0, 1, 0, 0,  1,
                 0.5,  0.5,  0.5, 0, 0, 0, 0,  1,

                -0.5, -0.5,  0.5, 1, 0, -1, 0,  0,
                -0.5,  0.5,  0.5, 1, 1, -1, 0,  0,
                -0.5,  0.5, -0.5, 0, 1, -1, 0,  0,

                -0.5,  0.5, -0.5, 0, 1, -1, 0,  0,
                -0.5, -0.5, -0.5, 0, 0, -1, 0,  0,
                -0.5, -0.5,  0.5, 1, 0, -1, 0,  0,

                 0.5, -0.5, -0.5, 1, 0, 1, 0,  0,
                 0.5,  0.5, -0.5, 1, 1, 1, 0,  0,
                 0.5,  0.5,  0.5, 0, 1, 1, 0,  0,

                 0.5,  0.5,  0.5, 0, 1, 1, 0,  0,
                 0.5, -0.5,  0.5, 0, 0, 1, 0,  0,
                 0.5, -0.5, -0.5, 1, 0, 1, 0,  0,

                 0.5, -0.5,  0.5, 0, 1, 0, -1,  0,
                -0.5, -0.5,  0.5, 1, 1, 0, -1,  0,
                -0.5, -0.5, -0.5, 1, 0, 0, -1,  0,

                -0.5, -0.5, -0.5, 1, 0, 0, -1,  0,
                 0.5, -0.5, -0.5, 0, 0, 0, -1,  0,
                 0.5, -0.5,  0.5, 0, 1, 0, -1,  0,

                 0.5,  0.5, -0.5, 0, 1, 0, 1,  0,
                -0.5,  0.5, -0.5, 1, 1, 0, 1,  0,
                -0.5,  0.5,  0.5, 1, 0, 0, 1,  0,

                -0.5,  0.5,  0.5, 1, 0, 0, 1,  0,
                 0.5,  0.5,  0.5, 0, 0, 0, 1,  0,
                 0.5,  0.5, -0.5, 0, 1, 0, 1,  0
            )
        

        self.vertex_count = len(self.vertices)//8
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

    def update(self):
        glUseProgram(self.shader)
        angle = np.radians(45)
        
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(model_transform, pyrr.matrix44.create_from_z_rotation(theta=angle,dtype=np.float32))
        model_transform = pyrr.matrix44.multiply(model_transform, pyrr.matrix44.create_from_translation(vec=np.array(self.position),dtype=np.float32))
        glUniformMatrix4fv(glGetUniformLocation(self.shader,"model"),1,GL_FALSE,model_transform)

    def draw(self):
        
        glUseProgram(self.shader)
        self.material.use()
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)


class Flag:
    def __init__(self, shader, material, position):
        self.material = material
        self.shader = shader
        self.position = position
        glUseProgram(shader)
        # x, y, z, s, t, nx, ny, nz
        
        self.vertices, self.vertices = ObjLoader.load_model("models/cloth.obj")
        self.vertex_count = len(self.vertices)//8
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

    def update(self):

        glUseProgram(self.shader)
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        
        #model_transform = pyrr.matrix44.multiply(model_transform, pyrr.matrix44.create_from_y_rotation(theta=angle,dtype=np.float32))
        model_transform = pyrr.matrix44.multiply(model_transform, pyrr.matrix44.create_from_translation(vec=np.array(self.position),dtype=np.float32))

        #updates flag sine animation
        t = (pg.time.get_ticks()/1000)
        glUniform1f(glGetUniformLocation(self.shader, "time"), t)

        glUniformMatrix4fv(glGetUniformLocation(self.shader,"model"),1,GL_FALSE,model_transform)

    def draw(self):
        
        glUseProgram(self.shader)
        self.material.use()
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)


class Flagpole:
    def __init__(self, position, model):
        self.model = model
        self.position = position 

    def draw(self):
        self.model.draw(self.position)


class skyBox:
    def __init__(self, model):
        self.model = model

    def draw(self,position):
        self.model.draw(position)

class Player:
    def __init__(self, position):
        self.position = np.array(position,dtype=np.float32)
        self.forward = np.array([0, 0, 0],dtype=np.float32)
        self.theta = -90
        self.phi = 0
        self.moveSpeed = 1
        self.global_up = np.array([0, 0, 1], dtype=np.float32)

    def move(self, direction, amount):
        walkDirection = (direction + self.theta) % 360
        self.position[0] += amount * self.moveSpeed * np.cos(np.radians(walkDirection),dtype=np.float32)
        self.position[1] += amount * self.moveSpeed * np.sin(np.radians(walkDirection),dtype=np.float32)

    def increment_direction(self, theta_increase, phi_increase):
        self.theta = (self.theta + theta_increase) % 360
        self.phi = min(max(self.phi + phi_increase,-89),89)

    def update(self,shaders):
        camera_cos = np.cos(np.radians(self.theta),dtype=np.float32)
        camera_sin = np.sin(np.radians(self.theta),dtype=np.float32)
        camera_cos2 = np.cos(np.radians(self.phi),dtype=np.float32)
        camera_sin2 = np.sin(np.radians(self.phi),dtype=np.float32)
        self.forward[0] = camera_cos * camera_cos2
        self.forward[1] = camera_sin * camera_cos2
        self.forward[2] = camera_sin2
        
        right = pyrr.vector3.cross(self.global_up,self.forward)
        up = pyrr.vector3.cross(self.forward,right)
        lookat_matrix = pyrr.matrix44.create_look_at(self.position, self.position + self.forward, up,dtype=np.float32)
        for shader in shaders:
            glUseProgram(shader)
            glUniformMatrix4fv(glGetUniformLocation(shader,"view"),1,GL_FALSE,lookat_matrix)
            glUniform3fv(glGetUniformLocation(shader,"cameraPos"),1,self.position)


class Ground:
    def __init__(self, position, model):
        self.model = model
        self.position = position

    def draw(self):
        self.model.draw(self.position)


class Light:
    def __init__(self, shaders, colour, position, strength, index):
        self.model = CubeBasic(shaders[0], 0.1, 0.1, 0.1, colour[0], colour[1], colour[2])
        self.colour = np.array(colour, dtype=np.float32)
        self.shaders = []
        for i in range(1,len(shaders)):
            self.shaders.append(shaders[i])
        self.position = np.array(position, dtype=np.float32)
        self.strength = strength
        self.index = index

    def update(self):
        for shader in self.shaders:
            glUseProgram(shader)
            glUniform3fv(glGetUniformLocation(shader,f"lights[{self.index}].pos"),1,self.position)
            glUniform3fv(glGetUniformLocation(shader,f"lights[{self.index}].color"),1,self.colour)
            glUniform1f(glGetUniformLocation(shader,f"lights[{self.index}].strength"),self.strength)
            glUniform1i(glGetUniformLocation(shader,f"lights[{self.index}].enabled"),1)

    def draw(self):
        self.model.draw(self.position)



### View
class Material:
    def __init__(self, filepath):
        self.diffuseTexture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.diffuseTexture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pg.image.load(f"{filepath}_diffuse.jpg").convert()
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        
        self.specularTexture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.specularTexture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pg.image.load(f"{filepath}_specular.jpg").convert()
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.diffuseTexture)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D,self.specularTexture)


class SimpleMaterial:
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pg.image.load(f"{filepath}.png").convert_alpha()
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.texture)


class CubeMapMaterial:
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        #load textures
        image = pg.image.load(f"{filepath}_left.png").convert_alpha()
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        image = pg.image.load(f"{filepath}_right.png").convert_alpha()
        image = pg.transform.flip(image,True,True)
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        image = pg.image.load(f"{filepath}_top.png").convert_alpha()
        image = pg.transform.rotate(image, 90)
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        image = pg.image.load(f"{filepath}_bottom.png").convert_alpha()
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        image = pg.image.load(f"{filepath}_back.png").convert_alpha()
        image = pg.transform.rotate(image, -90)
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        image = pg.image.load(f"{filepath}_front.png").convert_alpha()
        image = pg.transform.rotate(image, 90)
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP,self.texture)


class CubeBasic:
    def __init__(self, shader, l, w, h, r, g, b):
        self.shader = shader
        glUseProgram(shader)
        # x, y, z, r, g, b
        self.vertices = (
                -l/2, -w/2, -h/2, r, g, b,
                -l/2,  w/2, -h/2, r, g, b,
                 l/2,  w/2, -h/2, r, g, b,

                 l/2,  w/2, -h/2, r, g, b,
                 l/2, -w/2, -h/2, r, g, b,
                -l/2, -w/2, -h/2, r, g, b,

                 l/2,  w/2,  h/2, r, g, b,
                -l/2,  w/2,  h/2, r, g, b,
                -l/2, -w/2,  h/2, r, g, b,

                -l/2, -w/2,  h/2, r, g, b,
                 l/2, -w/2,  h/2, r, g, b,
                 l/2,  w/2,  h/2, r, g, b,

                -l/2, -w/2,  h/2, r, g, b,
                -l/2,  w/2,  h/2, r, g, b,
                -l/2,  w/2, -h/2, r, g, b,

                -l/2,  w/2, -h/2, r, g, b,
                -l/2, -w/2, -h/2, r, g, b,
                -l/2, -w/2,  h/2, r, g, b,

                 l/2, -w/2, -h/2, r, g, b,
                 l/2,  w/2, -h/2, r, g, b,
                 l/2,  w/2,  h/2, r, g, b,

                 l/2,  w/2,  h/2, r, g, b,
                 l/2, -w/2,  h/2, r, g, b,
                 l/2, -w/2, -h/2, r, g, b,

                 l/2, -w/2,  h/2, r, g, b,
                -l/2, -w/2,  h/2, r, g, b,
                -l/2, -w/2, -h/2, r, g, b,

                -l/2, -w/2, -h/2, r, g, b,
                 l/2, -w/2, -h/2, r, g, b,
                 l/2, -w/2,  h/2, r, g, b,

                 l/2,  w/2, -h/2, r, g, b,
                -l/2,  w/2, -h/2, r, g, b,
                -l/2,  w/2,  h/2, r, g, b,

                -l/2,  w/2,  h/2, r, g, b,
                 l/2,  w/2,  h/2, r, g, b,
                 l/2,  w/2, -h/2, r, g, b
            )
        self.vertex_count = len(self.vertices)//6
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    def draw(self, position):
        glUseProgram(self.shader)
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(model_transform, pyrr.matrix44.create_from_translation(vec=position,dtype=np.float32))
        glUniformMatrix4fv(glGetUniformLocation(self.shader,"model"),1,GL_FALSE,model_transform)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)


# obj loading
class ObjModel:
    def __init__(self, folderpath, filename, shader, material):
        self.shader = shader
        self.material = material
        glUseProgram(shader)
        v = []
        vt = []
        vn = []
        self.vertices = []

        #open the obj file and read the data
        with open(f"{folderpath}/{filename}",'r') as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                if flag=="mtllib":
                    pass
                elif flag=="v":
                    #vertex
                    line = line.replace("v ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                   
                elif flag=="vt":
                    line = line.replace("vt ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vt.append(l)
                elif flag=="vn":
                    line = line.replace("vn ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vn.append(l)
                elif flag=="f":
                    
                    line = line.replace("f ","")
                    line = line.replace("\n","")
                    line = line.split(" ")
                    theseVertices = []
                    theseTextures = []
                    theseNormals = []
                    for vertex in line:
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        theseVertices.append(v[position])
                        texture = int(l[1]) - 1
                        theseTextures.append(vt[texture])
                        normal = int(l[2]) - 1
                        theseNormals.append(vn[normal])
                   
                   
                    triangles_in_face = len(line) - 2

                    vertex_order = []
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i+1)
                        vertex_order.append(i+2)
                    for i in vertex_order:
                        for x in theseVertices[i]:
                            self.vertices.append(x)
                        for x in theseTextures[i]:
                            self.vertices.append(x)
                        for x in theseNormals[i]:
                            self.vertices.append(x)
                    
                line = f.readline()
        self.vertices = np.array(self.vertices,dtype=np.float32)

        #vertex array object, all that stuff
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER,self.vbo)
        glBufferData(GL_ARRAY_BUFFER,self.vertices.nbytes,self.vertices,GL_STATIC_DRAW)
        self.vertexCount = int(len(self.vertices)/8)

        #position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,self.vertices.itemsize*8,ctypes.c_void_p(0))
        #normal attribute
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,self.vertices.itemsize*8,ctypes.c_void_p(12))
        #texture attribute
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,self.vertices.itemsize*8,ctypes.c_void_p(20))

    def getVertices(self):
        return self.vertices

    def draw(self, position):
        glUseProgram(self.shader)
        self.material.use()
        model_transform = pyrr.matrix44.create_from_translation(position, dtype=np.float32)
        glUniformMatrix4fv(glGetUniformLocation(self.shader,"model"),1,GL_FALSE,model_transform)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES,0,self.vertexCount)
    

class TexturedQuad:
    def __init__(self, x, y, w, h, textures, shader):
        self.shader = shader
        self.textures = textures
        self.vertices = (
            x - w/2, y + h/2, 0, 1,
            x - w/2, y - h/2, 0, 0,
            x + w/2, y - h/2, 1, 0,

            x - w/2, y + h/2, 0, 1,
            x + w/2, y - h/2, 1, 0,
            x + w/2, y + h/2, 1, 1
        )
        self.vertices = np.array(self.vertices, dtype=np.float32)
        
        glUseProgram(self.shader)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
    
    def draw(self):
        glUseProgram(self.shader)
        for i in range(len(self.textures)):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, self.textures[i])
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES,0,6)
    

class BillBoard:
    def __init__(self, w, h, texture, shader):
        self.texture = texture
        self.shader = shader
        self.vertices = (
            0, -w/2,  h/2, 0, 1, -1, 0, 0,
            0, -w/2, -h/2, 0, 0, -1, 0, 0,
            0,  w/2, -h/2, 1, 0, -1, 0, 0,

            0, -w/2,  h/2, 0, 1, -1, 0, 0,
            0,  w/2, -h/2, 1, 0, -1, 0, 0,
            0,  w/2,  h/2, 1, 1, -1, 0, 0
        )
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vertexCount = 6
        
        glUseProgram(self.shader)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
    
    def draw(self, position, playerPosition):
        glUseProgram(self.shader)
        self.texture.use()
        directionFromPlayer = position - playerPosition
        angle1 = np.arctan2(-directionFromPlayer[1],directionFromPlayer[0])
        dist2d = math.sqrt(directionFromPlayer[0] ** 2 + directionFromPlayer[1] ** 2)
        angle2 = np.arctan2(directionFromPlayer[2],dist2d)

        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(model_transform,
            pyrr.matrix44.create_from_y_rotation(theta=angle2, dtype=np.float32))
        model_transform = pyrr.matrix44.multiply(model_transform,
            pyrr.matrix44.create_from_z_rotation(theta=angle1, dtype=np.float32))
        model_transform = pyrr.matrix44.multiply(model_transform,
                            pyrr.matrix44.create_from_translation(position,dtype=np.float32))
        glUniformMatrix4fv(glGetUniformLocation(self.shader,"model"),1,GL_FALSE,model_transform)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertexCount)

class CubeMapModel:
    def __init__(self, shader, l, w, h, r, g, b, material):
        self.material = material
        self.shader = shader
        glUseProgram(shader)
        # x, y, z, r, g, b
        self.vertices = (
                -l/2, -w/2, -h/2, r, g, b,
                -l/2,  w/2, -h/2, r, g, b,
                 l/2,  w/2, -h/2, r, g, b,

                 l/2,  w/2, -h/2, r, g, b,
                 l/2, -w/2, -h/2, r, g, b,
                -l/2, -w/2, -h/2, r, g, b,

                 l/2,  w/2,  h/2, r, g, b,
                -l/2,  w/2,  h/2, r, g, b,
                -l/2, -w/2,  h/2, r, g, b,

                -l/2, -w/2,  h/2, r, g, b,
                 l/2, -w/2,  h/2, r, g, b,
                 l/2,  w/2,  h/2, r, g, b,

                -l/2, -w/2,  h/2, r, g, b,
                -l/2,  w/2,  h/2, r, g, b,
                -l/2,  w/2, -h/2, r, g, b,

                -l/2,  w/2, -h/2, r, g, b,
                -l/2, -w/2, -h/2, r, g, b,
                -l/2, -w/2,  h/2, r, g, b,

                 l/2, -w/2, -h/2, r, g, b,
                 l/2,  w/2, -h/2, r, g, b,
                 l/2,  w/2,  h/2, r, g, b,

                 l/2,  w/2,  h/2, r, g, b,
                 l/2, -w/2,  h/2, r, g, b,
                 l/2, -w/2, -h/2, r, g, b,

                 l/2, -w/2,  h/2, r, g, b,
                -l/2, -w/2,  h/2, r, g, b,
                -l/2, -w/2, -h/2, r, g, b,

                -l/2, -w/2, -h/2, r, g, b,
                 l/2, -w/2, -h/2, r, g, b,
                 l/2, -w/2,  h/2, r, g, b,

                 l/2,  w/2, -h/2, r, g, b,
                -l/2,  w/2, -h/2, r, g, b,
                -l/2,  w/2,  h/2, r, g, b,

                -l/2,  w/2,  h/2, r, g, b,
                 l/2,  w/2,  h/2, r, g, b,
                 l/2,  w/2, -h/2, r, g, b
            )
        self.vertex_count = len(self.vertices)//6
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    def draw(self, position):
        glUseProgram(self.shader)
        self.material.use()
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(model_transform, pyrr.matrix44.create_from_translation(vec=position,dtype=np.float32))
        glUniformMatrix4fv(glGetUniformLocation(self.shader,"model"),1,GL_FALSE,model_transform)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)


### Control
class Scene:
    def __init__(self):
        global shaders
        global framebuffer

        self.shader3DTextured = shaders[0]
        self.shader3DColored = shaders[1]
        self.shader2DTextured = shaders[2]
        self.shader3DBillboard = shaders[6]
        self.shader3DCubemap = shaders[7]
        self.shader3DCloth = shaders[8]

        self.multisampleFBO = framebuffer[0]
        self.regularCBMultisampled = framebuffer[1]
        self.brightCBMultisampled = framebuffer[2]
        self.MultisampleDepthStencilBuffer = framebuffer[3]
        self.singlesampleFBO = framebuffer[4]
        self.regularCB = framebuffer[5]
        self.brightCB = framebuffer[6]

        pg.mouse.set_visible(False)
        self.lastTime = 0
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0
        self.lightCount = 0
        self.resetLights()
        self.create_objects()

    def create_objects(self):
        self.wood_texture = Material("gfx/crate")
        self.sand_texture = Material("gfx/sand")
        self.cloth_texture = Material("gfx/eye")  
        self.metal_texture = Material("gfx/metal")
        self.pyramid_texture = Material("gfx/hiero")  
 
        self.cube = Cube(self.shader3DTextured, self.wood_texture, [2,1,0.8])
        self.flag = Flag(self.shader3DCloth, self.cloth_texture,[-0.61,0,3.2])
        flagpole_model = ObjModel("models", "flagpole.obj", self.shader3DTextured, self.metal_texture)
        pyramid_model = ObjModel("models", "pyramid.obj", self.shader3DTextured, self.pyramid_texture)
        self.pyramid = Flagpole(np.array([-2,1,0.3],dtype=np.float32), pyramid_model)
        self.flagpole = Flagpole(np.array([0,0,2.1],dtype=np.float32), flagpole_model)
        

        self.player = Player([0,4,2])
        self.light = Light([self.shader3DColored, self.shader3DTextured,self.shader3DBillboard], [0.2, 0.7, 0.8], [1,1.7,3], 2, self.lightCount)
        self.lightCount += 1
        
        self.light2 = Light([self.shader3DColored, self.shader3DTextured, self.shader3DBillboard], [0.9, 0.4, 0.0], [0,1.7,3], 2, self.lightCount)
        self.lightCount += 1

        self.screen = TexturedQuad(0, 0, 2, 2, (self.regularCB, self.brightCB), self.shader2DTextured)
        ground_model = ObjModel("models", "rock.obj", self.shader3DTextured, self.sand_texture)
        self.ground = Ground(np.array([0,0,0],dtype=np.float32), ground_model)
        
        self.skyBoxTexture = CubeMapMaterial("gfx/sky")
        skyBoxModel = CubeMapModel(self.shader3DCubemap, 10,10,10,1,1,1, self.skyBoxTexture)
        self.skyBox = skyBox(skyBoxModel)

    def resetLights(self):
        glUseProgram(self.shader3DBillboard)
        for i in range(8):
            glUniform1i(glGetUniformLocation(self.shader3DBillboard,f"lights[{i}].enabled"),0)
        glUseProgram(self.shader3DTextured)
        for i in range(8):
            glUniform1i(glGetUniformLocation(self.shader3DTextured,f"lights[{i}].enabled"),0)

    def mainLoop(self):
        result = CONTINUE
        #check events
        for event in pg.event.get():
            if (event.type == pg.KEYDOWN and event.key==pg.K_ESCAPE):
                result = EXIT
                
        self.handleMouse()
        self.handleKeys()
        
        #update objects
        self.cube.update()
        self.flag.update()
        self.light.update()
        self.light2.update()
        self.player.update([self.shader3DColored, self.shader3DTextured, self.shader3DBillboard, self.shader3DCubemap, self.shader3DCloth])

        #first pass
        glBindFramebuffer(GL_FRAMEBUFFER, self.multisampleFBO)
        glDrawBuffers(2, (GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        self.skyBox.draw(self.player.position)
        glEnable(GL_CULL_FACE)
        self.cube.draw()
        self.flag.draw()
        self.pyramid.draw()
        self.flagpole.draw()
        self.light.draw()
        self.light2.draw()
        self.ground.draw()        

        #bounce multisampled frame down to single sampled
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.multisampleFBO)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.singlesampleFBO)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        glDrawBuffer(GL_COLOR_ATTACHMENT0)
        glBlitFramebuffer(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GL_COLOR_BUFFER_BIT, GL_NEAREST)
        glReadBuffer(GL_COLOR_ATTACHMENT1)
        glDrawBuffer(GL_COLOR_ATTACHMENT1)
        glBlitFramebuffer(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GL_COLOR_BUFFER_BIT, GL_NEAREST)

        #second pass
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        self.screen.draw()
        
        pg.display.flip()
        #timing
        self.showFrameRate()
        return result

    def handleKeys(self):
        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            self.player.move(0, 0.0025*self.frameTime)
            return
        if keys[pg.K_a]:
            self.player.move(90, 0.0025*self.frameTime)
            return
        if keys[pg.K_s]:
            self.player.move(180, 0.0025*self.frameTime)
            return
        if keys[pg.K_d]:
            self.player.move(-90, 0.0025*self.frameTime)
            return

    def handleMouse(self):
        (x,y) = pg.mouse.get_pos()
        theta_increment = self.frameTime * 0.2 * (SCREEN_WIDTH / 2 - x)
        phi_increment = self.frameTime * 0.2 * (SCREEN_HEIGHT / 2 - y)
        self.player.increment_direction(theta_increment, phi_increment)
        pg.mouse.set_pos((SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))

    def showFrameRate(self):
        self.currentTime = pg.time.get_ticks()
        delta = self.currentTime - self.lastTime
        if (delta >= 1000):
            framerate = int(1000.0 * self.numFrames/delta)
            pg.display.set_caption(f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / max(60,framerate))
        self.numFrames += 1


class startScene:
    def __init__(self):
        global shaders

        self.shader2DColored = shaders[3]
        self.shaderText = shaders[4]
        self.particleShader = shaders[5]

        pg.mouse.set_visible(True)
        self.lastTime = 0
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0
        
        

    def mainLoop(self):
        result = NEW_SCENE
        #check events
        for event in pg.event.get():
            if (event.type == pg.MOUSEBUTTONDOWN and event.button==1):
                result = self.handleMouseClick()
            if (event.type == pg.KEYDOWN and event.key==pg.K_ESCAPE):
                result = EXIT
        self.handleMouseMove()
        
        #render
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)

        #timing
        self.showFrameRate()
        return result

    def handleMouseMove(self):
        (x,y) = pg.mouse.get_pos()
        x -= SCREEN_WIDTH / 2
        x /= SCREEN_WIDTH / 2
        y -= SCREEN_HEIGHT / 2
        y /= -SCREEN_HEIGHT / 2

    
    def handleMouseClick(self):
        (x,y) = pg.mouse.get_pos()
        x -= SCREEN_WIDTH / 2
        x /= SCREEN_WIDTH / 2
        y -= SCREEN_HEIGHT / 2
        y /= -SCREEN_HEIGHT / 2

        CONTINUE = 1
        return CONTINUE

    def showFrameRate(self):
        self.currentTime = pg.time.get_ticks()
        delta = self.currentTime - self.lastTime
        if (delta >= 1000):
            framerate = int(1000.0 * self.numFrames/delta)
            pg.display.set_caption(f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / max(60,framerate))
        self.numFrames += 1



# scene initiation
initialise_pygame()
shaders = initialise_opengl()
framebuffer = create_framebuffer()
starter = startScene()
result = CONTINUE
while(result == CONTINUE):
    result = starter.mainLoop()
    if result == NEW_SCENE:
        starter = Scene()
        result = CONTINUE
        continue
    
