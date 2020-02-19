#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "EasyBMP/EasyBMP.h"

using namespace std;

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

#define WINDOW_SIZE 3
#define WINDOW_LENGHT WINDOW_SIZE *WINDOW_SIZE

texture<float, cudaTextureType2D, cudaReadModeElementType> tex;

__global__ void saltAndPepperWithCuda(float* output, int imageWidth, int imageHeight)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (!(row < imageHeight && col < imageWidth)) {
        return;
    }

    float filter[WINDOW_LENGHT] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int x = 0; x < WINDOW_SIZE; x++)
    {
        for (int y = 0; y < WINDOW_SIZE; y++)
        {
            filter[x * WINDOW_SIZE + y] = tex2D(tex, col + y - 1, row + x - 1);
        }
    }
    for (int i = 0; i < WINDOW_LENGHT; i++)
    {
        for (int j = i + 1; j < WINDOW_LENGHT; j++)
        {
            if (filter[i] > filter[j])
            {
                float tmp = filter[i];
                filter[i] = filter[j];
                filter[j] = tmp;
            }
        }
    }
    output[row * imageWidth + col] = filter[(int)(WINDOW_LENGHT / 2)];
}

float* readLikeGrayScale(char* filePathInput, unsigned int* rows, unsigned int* cols);
void writeImage(char* filePath, float* grayscale, unsigned int rows, unsigned int cols);

int main()
{
    float* grayscale = 0;
    unsigned int rows, cols;

    grayscale = readLikeGrayScale("images.bmp", &rows, &cols);
    writeImage("afterRead.bmp", grayscale, rows, cols);
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0,
            cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, cols, rows));

    checkCudaErrors(cudaMemcpyToArray(cuArray, 0, 0, grayscale, rows * cols * sizeof(float),
        cudaMemcpyHostToDevice));

    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;

    checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));

    float* dev_output, * output;
    output = (float*)calloc(rows * cols, sizeof(float));
    cudaMalloc(&dev_output, rows * cols * sizeof(float));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
        (rows + dimBlock.y - 1) / dimBlock.y);
    saltAndPepperWithCuda << <dimGrid, dimBlock >> > (dev_output, cols, rows);
    checkCudaErrors(cudaMemcpy(output, dev_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    writeImage("result.bmp", output, rows, cols);
    cudaFreeArray(cuArray);
    cudaFree(dev_output);
    return 0;
}

float* readLikeGrayScale(char* filePathInput, unsigned int* rows, unsigned int* cols)
{
    BMP Input;
    Input.ReadFromFile(filePathInput);
    *rows = Input.TellHeight();
    *cols = Input.TellWidth();
    float* grayscale = (float*)calloc(*rows * *cols, sizeof(float));
    for (int j = 0; j < *rows; j++)
    {
        for (int i = 0; i < *cols; i++)
        {
            float gray = (float)floor(0.299 * Input(i, j)->Red +
                0.587 * Input(i, j)->Green +
                0.114 * Input(i, j)->Blue);
            grayscale[j * *cols + i] = gray;
        }
    }
    return grayscale;
}

void writeImage(char* filePath, float* grayscale, unsigned int rows, unsigned int cols)
{
    BMP Output;
    Output.SetSize(cols, rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            RGBApixel pixel;
            pixel.Red = grayscale[i * cols + j];
            pixel.Green = grayscale[i * cols + j];
            pixel.Blue = grayscale[i * cols + j];
            pixel.Alpha = 0;
            Output.SetPixel(j, i, pixel);
        }
    }
    Output.WriteToFile(filePath);
}