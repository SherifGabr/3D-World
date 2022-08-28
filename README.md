# 3D World Animation
3D animation of flag and some objects using Python, PyGame, and OpenGL. The world consists of a ground and a cubemap in addition to two lights.

# 1: Modeling, Blender, and Importing
Screenshot of Blender’s UI. I designed the flag and flagpole in Blender. The flag is a cube whose dimensions were altered so that it resembles the flag cloth.


In the screenshot, they have their coordinates zeroed so that they retain their zero position after they are loaded in OpenGL later on. Later, they were exported as separate .obj files. They are ready to be imported into Python/OpenGL.
![Blender](https://user-images.githubusercontent.com/20493629/185512808-cceda3fc-eda5-4d51-a7c2-d21b094f3d01.png)


Using the Objloader file from the original assignments, I loaded the flag into its class Flag and used its separate animation shader shader3DCloth. 

# 2: Flag Animation
As outlined in the .pdf as well as from some sources online, a sine wave was used for the animation. It was implemented in the vertex shader. The parameters were selected based on trial an error. The values are as follows:
 
A = 0.5;
k = 4;
omega = 5;
phi = 4;

The first issue that arised is that the left/right side of the flag is not attached to the flagpole. This was solved by multiplying sine wave 1 by another sine wave that is shifted slightly. The created a stationary point on the side of the flag near to the flagpole.

float y = A*vertexPos[0]*sin(k*vertexPos[0] - omega*time + phi); 
y = y * sin(vertexPos[0] - 10);	

The time variable in the vertex shader to a variable t in Python. Time information is retrieved from the PyGame library. It updates the time value of the sine wave with each frame.

# 3: Camera and keyboard controls
The current functionality provides rotation in all degrees of freedom for the camera whereas the translation is limited to front, backwards, right, and left. It is implemented by the Player class. 
Rotation is controlled by the mouse movement (handleMouse function). The function updates theta (x-axis/yaw) and phi (y-axis/pitch) by multiplying a step value (in this case 0.2) by the frameTime attribute.
Translation was made possible by handleKeys function which is activated by key presses (pg.key.get_pressed()). If the key matches one of w, a, s, or d keys, motion starts in that direction by multiplying a step value (in this case 0.0025) by the frameTime attribute.

# 4: Gouraud shading
For this point, I was not able/did not have the time capacity to finish it. There are some useful tutorials but phong shading was the more popular topic in the shading tutorials. 

# 5: Skymap using cube mapping 

This was done by instantiating the class CubeMapBasicMaterial. Its working principle is by using 5 images of a sky image and projecting each image to a face of the cube using the GL_TEXTURE_CUBE_MAP_{DIRECTION_DIMENSION}.


# 6: Realism
To improve realism, the object materials contain diffuse and specular channels. This is carried out by the Material class which takes an image as input for each of the two channels: diffuse and specular. Also real life textures were used instead of solid colors in most of the objects.

# Final Output

Scene of flag and two objects: a box and a pyramid. FPS is shown in the title bar. It proved to be a useful feature for debugging as it would either be 0 or very high sometimes which often alerted me that there was something wrong with the implementation. It averages about 150 - 250 FPS.
![Final](https://user-images.githubusercontent.com/20493629/185512825-4a05f84b-1279-493e-8d01-dc7cb917cde1.png)

# Sources/References

https://github.com/amengede/getIntoGameDev/tree/main/pyopengl
https://www.youtube.com/watch?v=LCK1qdp_HhQ&list=PLn3eTxaOtL2PDnEVNwOgZFm5xYPr4dUoR&ab_channel=GetIntoGameDev
https://www.youtube.com/playlist?list=PLQVvvaa0QuDdfGpqjkEJSeWKGCP31__wD
All models were created in Blender except for the pyramid it was provided by www.cgtrader.com
