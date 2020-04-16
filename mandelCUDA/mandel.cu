#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include <sys/time.h>

#include <SDL2/SDL.h>

#define WINDOW_WIDTH 1024
#define ITRS 20000

__global__
void mandelCUDA(float *xa, float *ya, float *muResults, int maxIters) {
	float xn = 0, x = 0, y = 0;
	unsigned int iters = 0, gid = threadIdx.x + blockDim.x*blockIdx.x;

	float xp = xa[gid];
	float yp = ya[gid];

	while((x*x + y*y < (float)5.0) && (iters < maxIters))
	{
		xn = x*x - y*y + xp;
		y = (float)2.0*x*y + yp;
		x = xn;
		iters++;
	}

	if(iters >= maxIters) {
		muResults[gid] = maxIters;
	} else {
		xn = x*x - y*y + xp;
		y = (float)2.0*x*y + yp;
		x = xn;

		xn = x*x - y*y + xp;
		y = (float)2.0*x*y + yp;
		x = xn;

		muResults[gid] = (iters+2) - logf(0.5*logf(x*x+y*y))/logf(2.0);
	}
}

float mandel(float xp, float yp, int maxIters);

void iters2rgb(float mu, float muMax, int *r, int *g, int *b);

int main(void) {
	struct timeval start, end;
	SDL_Event event;
	SDL_Renderer *renderer;
	SDL_Window *window, *surfWindow;
	SDL_Surface *surface;

	// Set up SDL, create window, etc.
	SDL_Init(SDL_INIT_VIDEO);

	surfWindow = SDL_CreateWindow("Surface Window", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_WIDTH, 0);
	surface = SDL_GetWindowSurface(surfWindow);
	printf("%08x %08x %08x %08x %d\n", surface->format->Rmask, surface->format->Gmask,
			surface->format->Bmask, surface->format->Amask, surface->format->BitsPerPixel);

	//SDL_UpdateWindowSurface(window); // Run this after writing to surface->pixels

	// Example of how to draw a pixel
	//    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Set colour in RGBA formal
	//    SDL_RenderDrawPoint(renderer, 0, 0); // Apply colour to a pixel
	//   SDL_RenderPresent(renderer); // Update the window

	int px = 0, py = 0;

	// Allocate GPU memory
	// Lists of cmplx numbers for computation
	float *x, *y;
	cudaMallocManaged(&x, WINDOW_WIDTH*WINDOW_WIDTH*sizeof(float));
	cudaMallocManaged(&y, WINDOW_WIDTH*WINDOW_WIDTH*sizeof(float));

	float *muResults;
	cudaMallocManaged(&muResults, WINDOW_WIDTH*WINDOW_WIDTH*sizeof(float));

	//float top = 1.5, bottom = -1.5, left = -2.0, right = 1;
	float zoom = 1.5, cx = -0.75, cy = 0;
	while(1) {
		// Fills in the x-y complex co-ordinates for each CUDA thread
		for(py = 0; py < WINDOW_WIDTH; py++)
		{
			for(px = 0; px < WINDOW_WIDTH; px++)
			{
				float xTmp = (float)2.0*zoom/WINDOW_WIDTH*px+(cx-zoom);
				float yTmp = -(float)2.0*zoom/WINDOW_WIDTH*py+(cy+zoom);
				x[px+WINDOW_WIDTH*py] = xTmp;
				y[px+WINDOW_WIDTH*py] = yTmp;
			}
		}

		//TODO: Call the GPU kernel
		gettimeofday(&start, 0);
		mandelCUDA<<<1024,1024>>>(x, y, muResults, ITRS);
		cudaDeviceSynchronize();
		gettimeofday(&end, 0);
		printf("Time: %f\n", (end.tv_sec + (float)end.tv_usec/1e6) - 
				(start.tv_sec + (float)start.tv_usec/1e6));
		// Copy results to SDL_Surface pixel data
		for(py = 0; py < WINDOW_WIDTH; py++)
		{
			for(px = 0; px < WINDOW_WIDTH; px++)
			{
				float mu = muResults[px+py*WINDOW_WIDTH];
				int r, g, b;
				iters2rgb(mu, ITRS, &r, &g, &b);
				if((int)mu == ITRS) {
					uint32_t *pix;
					pix = (uint32_t*)surface->pixels;
					pix[px+WINDOW_WIDTH*py] = (uint32_t)0;
				} else {
					uint32_t *pix;
					iters2rgb(mu, ITRS, &r, &g, &b);
					pix = (uint32_t*)surface->pixels;
					pix[px+WINDOW_WIDTH*py] = ((uint8_t)r << 16) |
						((uint8_t)g << 8) |
						((uint8_t)b);
				}
			}
		}
		SDL_UpdateWindowSurface(surfWindow);


		int changed = 0;
		while(SDL_WaitEvent(&event) && (changed == 0)) {
			if(event.type == SDL_QUIT) {
				SDL_DestroyRenderer(renderer);
				SDL_DestroyWindow(window);
				SDL_Quit();
				return EXIT_SUCCESS;
			}
			if(event.type == SDL_MOUSEBUTTONDOWN) {
				cx = (float)2.0*zoom/WINDOW_WIDTH*event.button.x+(cx-zoom);
				cy = -(float)2.0*zoom/WINDOW_WIDTH*event.button.y+(cy+zoom);
				if(event.button.button == SDL_BUTTON_LEFT)
					zoom *= 0.8;
				if(event.button.button == SDL_BUTTON_RIGHT)
					zoom /= 0.8;
				changed = 1;
				printf("Changed\n");
			}
		}
	}
	// Idle until window is closed
	while(event.type != SDL_QUIT) 
	    SDL_PollEvent(&event);

	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return EXIT_SUCCESS;
}

float mandel(float xp, float yp, int maxIters)
{
	float xn = 0, x = 0, y = 0;
	unsigned int iters = 0;

	while((x*x + y*y < (float)5.0) && (iters < maxIters))
	{
		xn = x*x - y*y + xp;
		y = (float)2.0*x*y + yp;
		x = xn;
		iters++;
	}
	if(iters >= maxIters)
		return iters;

	xn = x*x - y*y + xp;
	y = (float)2.0*x*y + yp;
	x = xn;

	xn = x*x - y*y + xp;
	y = (float)2.0*x*y + yp;
	x = xn;

	float mu = (iters+2) - logf(0.5*logf(x*x+y*y))/logf(2.0);
	return mu;
}

void iters2rgb(float mu, float muMax, int *r, int *g, int *b)
{
//	float hue = (float)240.0*mu/muMax;

	float hue = log(mu)/log(muMax)*240.0;

	int H = hue/60;

	float residual = hue - H*60;

	switch(H) {
		case 0: *r = 255.0; *g = 255.0/60.0*residual; *b = 0; break;
		case 1: *r = 255.0-255.0/60.0*residual; *g = 255.0; *b = 0; break;
		case 2: *r = 0; *g = 255; *b = 255.0/60.0*residual; break;
		case 3: *r = 0; *g = 255.0-255.0/60.0*residual; *b = 255; break;
		case 4: *r = 255.0/60.0*residual; *g = 0; *b = 255;
	}
}
