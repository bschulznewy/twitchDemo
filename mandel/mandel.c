#include <omp.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>

#include <SDL2/SDL.h>

#define WINDOW_WIDTH 1000
#define ITRS 1000

float mandel(float xp, float yp, int maxIters);

void iters2rgb(float mu, float muMax, int *r, int *g, int *b);

int main(void) {
	SDL_Event event;
	SDL_Renderer *renderer;
	SDL_Window *window, *surfWindow;
	SDL_Surface *surface;
	int i;

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
	unsigned char pxData[WINDOW_WIDTH][WINDOW_WIDTH][3];

	//float top = 1.5, bottom = -1.5, left = -2.0, right = 1;
	float zoom = 1.5, cx = -0.75, cy = 0;;
	while(1) {
	float start = omp_get_wtime();
	#pragma omp parallel for schedule(dynamic) private(px) collapse(1)
	for(py = 0; py < WINDOW_WIDTH; py++)
	{
		for(px = 0; px < WINDOW_WIDTH; px++)
		{
			//float y = (float)(bottom-top)/(float)WINDOW_WIDTH*(float)py + (float)top;
			//float x = (float)(right-left)/(float)WINDOW_WIDTH*(float)px - (float)left;

			float x = (float)2.0*zoom/WINDOW_WIDTH*px+(cx-zoom);
			float y = -(float)2.0*zoom/WINDOW_WIDTH*py+(cy+zoom);

			float mu, muTmp;
			int r, g, b, cnt = 0;;
			mu = mandel(x, y, ITRS);
			cnt++;
			//muTmp = mandel(x+zoom/WINDOW_WIDTH, y+zoom/WINDOW_WIDTH, ITRS);
			//if(muTmp < ITRS){cnt++; mu+=muTmp;}
			muTmp = mandel(x+zoom/WINDOW_WIDTH, y, ITRS);
			if(muTmp < ITRS){cnt++; mu+=muTmp;}
			//muTmp = mandel(x+zoom/WINDOW_WIDTH, y-zoom/WINDOW_WIDTH, ITRS);
			//if(muTmp < ITRS){cnt++; mu+=muTmp;}
			muTmp = mandel(x, y-zoom/WINDOW_WIDTH, ITRS);
			if(muTmp < ITRS){cnt++; mu+=muTmp;}
			//muTmp = mandel(x-zoom/WINDOW_WIDTH, y-zoom/WINDOW_WIDTH, ITRS);
			//if(muTmp < ITRS){cnt++; mu+=muTmp;}
			muTmp = mandel(x-zoom/WINDOW_WIDTH, y, ITRS);
			//if(muTmp < ITRS){cnt++; mu+=muTmp;}
			//muTmp = mandel(x-zoom/WINDOW_WIDTH, y+zoom/WINDOW_WIDTH, ITRS);
			if(muTmp < ITRS){cnt++; mu+=muTmp;}
			muTmp = mandel(x, y+zoom/WINDOW_WIDTH, ITRS);
			if(muTmp < ITRS){cnt++; mu+=muTmp;}
			mu /= (float)cnt;
			if((int)mu == ITRS) {
				uint32_t *pix;
				pxData[px][py][0] = 0;
				pxData[px][py][1] = 0;
				pxData[px][py][2] = 0;
				pix = (uint32_t*)surface->pixels;
				pix[px+WINDOW_WIDTH*py] = (uint32_t)0;
				//SDL_SetRenderDrawColor(renderer, (int)0, (int)0, (int)0, 255);
			} else {
				uint32_t *pix;
				iters2rgb(mu, ITRS, &r, &g, &b);
				pxData[px][py][0] = r;
				pxData[px][py][1] = g;
				pxData[px][py][2] = b;
				pix = (uint32_t*)surface->pixels;
				pix[px+WINDOW_WIDTH*py] = ((uint8_t)r << 16) |
					((uint8_t)g << 8) |
					((uint8_t)b);
				///SDL_SetRenderDrawColor(renderer, r,g,b, 255);
			}

			//SDL_RenderDrawPoint(renderer, px, py); // Apply colour to a pixel
		}
		if(omp_get_thread_num() == 0)
			SDL_UpdateWindowSurface(surfWindow);
	}
	SDL_UpdateWindowSurface(surfWindow);
	printf("Time: %f\n", omp_get_wtime() - start);

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
//	return (float)iters;
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

