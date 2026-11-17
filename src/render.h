#pragma once
void initDeviceData(SceneData * sceneData_h, CameraData* cameraData_h, int width, int height);
void renderPass(color* sumColor, unsigned char* dst, int pass, int width, int height);
void destroy();



