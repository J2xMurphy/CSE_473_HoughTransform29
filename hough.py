import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import scipy.ndimage as sp
import copy
def main():
	img = cv2.imread('HoughCircles.jpg',0)
	print(img[0][0])
	toot = setup_zeros(img)
	forlorn = []
	points = innumerate(toot)
	tg = 0
	for tg in range(len(points)):
		lf = []
		nopts = 2+((len(points[tg])/32))
		print("Segment "+str(tg)+"/"+str(len(points)))
		pts2img(points[tg],img)
		for r in range(13,35):
			mm = minmax(points[tg])
			full = spatial(mm[1][1]-mm[0][1]+1,mm[1][0]-mm[0][0]+1)
			spps=[]
			for i in range(len(points[tg])):
				x = points[tg][i][0]-mm[0][0]
				y = points[tg][i][1]-mm[0][1]
				spps.append((x,y))
			for i in range(len(spps)/4):
				softcircle(spps[i*4][1],spps[i*4][0],r*2,full)
			for i in range(len(full[0])):
				for j in range(len(full)):
					if (10*nopts)<full[j][i]:
						n = (i+mm[0][1],j+mm[0][0],r*2)
						lf.append(n[:])
		forlorn+=fourlorn(lf)
	print(forlorn)
	full = [[0 for i in range(len(img[0]))]for j in range(len(img))]
	for k in range(len(forlorn)):
		y= forlorn[k][0]
		x= forlorn[k][1]
		r= forlorn[k][2]
		hardc(x,y,r,full)
	save(full,'ending')
	img = cv2.imread('HoughCircles.jpg')
	for i in range(len(full)):
		for j in range(len(full[i])):
			if full[i][j]==255:
				img[i][j]=np.asarray([255,0,0])
	save_color(img,'Beautiful')

def save_color(nplist,outfile):
	nplist = np.asarray(nplist,dtype='float32')
	#nplist = cv2.cvtColor(nplist,cv2.COLOR_BGR)
	cv2.imwrite(outfile+'.jpg',nplist)

def spatial(x,y):
	return [[0 for r in range(x)]for t in range(y)]

def minmax(points):
	lx=1000
	ly=1000
	mx=0
	my=0
	for i in range(len(points)):
		if points[i][0]<lx:
			lx=points[i][0]
		if points[i][0]>mx:
			mx=points[i][0]
		if points[i][1]<ly:
			ly=points[i][1]
		if points[i][1]>my:
			my=points[i][1]
	return[(lx,ly),(mx,my)]

def pts2img(points,img):
	full = [[0 for r in range(len(img[0]))]for t in range(len(img))]
	for i in range(len(points)):
		full[points[i][0]][points[i][1]]=255
	save(full,'Circles/tg')

def innumerate(img):
	final = []
	for i in range(2,len(img[0])-2):
		for j in range(2,len(img)-2):
			if img[j][i]>50:
				final.append((j,i))
	return final

def hardc(x,y,r,full):
	for i in range(360):
		if 0<int(x+r*math.cos(i))<len(full[0]) and 0<int(y+r*math.sin(i))<len(full):
			full[int(y+r*math.sin(i))][int(x+r*math.cos(i))]=255
	return full

def softcircle(x,y,r,full):
	addr = [[0 for i in range(len(full[0]))]for j in range(len(full))]
	for i in range(360):
		if 0<int(x+r*math.cos(i))<len(full[0]) and 0<int(y+r*math.sin(i))<len(full):
			addr[int(y+r*math.sin(i))][int(x+r*math.cos(i))]=10
	for i in range(len(full[0])):
		for j in range(len(full)):
			full[j][i]+=addr[j][i]
	return full

def closest(am, r):
	closest=255
	target=0
	print("LOL "+str((am,r)))
	for ele in r:
		if abs(ele-am)<closest:
			closest=abs(ele-am)
			target=ele
	print(closest)
	return target

def atmax(array):
	high = 0
	macx=0
	for i in range(len(array)):
		if array[i]>high:
			macx=i
			high = array[i]
	return macx

def zeroCB(img):
	print(len(img),len(img[0]))
	full = [[0 for i in range(len(img[0]))]for j in range(len(img))]
	for i in range(1,len(img)-1):
		for j in range(1,len(img[i])-1):
			count = 0
			if abs(img[i][j]-((img[i+1][j]+img[i-1][j]+img[i][j-1]+img[i][j-1])/4))>0:
				full[i][j]=255
			else:
				full[i][j]=0
	return full

def threshold(liste):
	for i in range(len(liste)):
		if liste[i]<liste[i+1]:
			return i
	return 0

def plot(img):
	array=[0 for i in range(0,256)]
	for i in range(len(img)):	
		for j in range(len(img[i])):
			array[img[i][j]]+=1
	arrayz=smooth(array)
	for i in range(50):
		arrayz=smooth(arrayz)
	plt.plot(arrayz)
	plt.ylabel('some numbers')
	th=threshold(array)
	return arrayz

def peaks(array):
	whe=[]
	whe.append(0)
	for i in range(1,len(array)-1):
		if array[i-1]<array[i]>array[i+1]:
			whe.append(i)
	whe.append(255)
	wha = []
	wha.append(0)
	for i in range(1,len(whe)):
		wha.append((whe[i-1]+whe[i])/2)
	wha.append(255)
	return wha

def smooth(lis):
	final=[]
	final.append((lis[0]+lis[1])/2)
	for i in range(1,len(lis)-1):
		final.append((lis[i]+lis[i+1]+lis[i-1])/3)
	final.append((lis[len(lis)-1]+lis[len(lis)-2])/2)
	return final

def save(nplist,outfile):
	nplist = np.asarray(nplist,dtype='float32')
	nplist = cv2.cvtColor(nplist,cv2.COLOR_GRAY2BGR)
	cv2.imwrite(outfile+'.jpg',nplist)

def similar(img,holds):
	thresh = 255/(len(holds)-2)
	print(thresh,len(holds))
	y = []
	full = [[0 for i in range(len(img[0]))]for j in range(len(img))]
	for i in range(1,len(img)-1):	
		for j in range(1,len(img[i])-1):
			for k in range(len(holds)-1):
				if holds[k]<img[i][j]<holds[k+1]:
					if thresh*k not in y:
						y.append(thresh*k)
					full[i][j]=thresh*k
	y.sort()
	return full,y

def innumerate(img):
	final = []
	for i in range(2,len(img[0])-2):
		for j in range(2,len(img)-2):
			if img[j][i]>50:
					final.append(localrelevant(j,i,img,[]))
	return final

def localrelevant(x,y,img,connected):
	img[x][y]=0
	connected.append((x,y))
	if (img[x+1][y]>50):
		connected = connected+localrelevant(x+1,y,img,[])

	if (img[x+1][y+1]>50):
		connected = connected+localrelevant(x+1,y+1,img,[])

	if (img[x+1][y-1]>50):
		connected = connected+localrelevant(x+1,y-1,img,[])

	if (img[x][y+1]>50):
		connected = connected+localrelevant(x,y+1,img,[])

	if (img[x][y-1]>50):
		connected = connected+localrelevant(x,y-1,img,[])

	if (img[x-1][y]>50):
		connected = connected+localrelevant(x-1,y,img,[])

	if (img[x-1][y+1]>50):
		connected = connected+localrelevant(x-1,y+1,img,[])

	if (img[x-1][y-1]>50):
		connected = connected+localrelevant(x-1,y-1,img,[])

	return connected

def setup_zeros(img):
	img = sp.gaussian_filter(img, sigma=3)
	array = plot(img)
	r=peaks(array)
	print(r)
	am =atmax(array)
	print(am)
	toot = similar(img,r)
	n = toot[1]
	toot=toot[0]
	print(n)
	sutin = closest(am,n)
	print("AS "+str(sutin))
	n=[0,sutin,255]
	toot = similar(img,n)
	n = toot[1]
	toot=toot[0]
	toot = zeroCB(toot)
	save(toot,'gr')
	return toot

def fourlorn(forlorn):
	fy = []
	fx = []
	fr = []
	for k in range(len(forlorn)):
		y= forlorn[k][1]
		x= forlorn[k][0]
		r= forlorn[k][2]
		fy.append(y)
		fx.append(x)
		fr.append(r)
	nfr=[]
	nfy=[]
	nfx=[]
	tfr=[]
	tfy=[]
	tfx=[]
	for i in range(len(fr)-1):
		if abs(fr[i]-fr[i+1])<6:
			tfr.append(fr[i])
			tfy.append(fy[i])
			tfx.append(fx[i])
		else:
			tfr.append(fr[i])
			tfy.append(fy[i])
			tfx.append(fx[i])
			nfr.append((np.sum(tfr))/len(tfr))
			nfy.append((np.sum(tfy))/len(tfy))
			nfx.append((np.sum(tfx))/len(tfx))
			tfr=[]
			tfy=[]
			tfx=[]
	if len(tfr)>0:
		nfr.append((np.sum(tfr))/len(tfr))
		nfy.append((np.sum(tfy))/len(tfy))
		nfx.append((np.sum(tfx))/len(tfx))
	final =[]
	for i in range(len(nfr)):
		final.append((nfy[i],nfx[i],nfr[i]))
	return final

def similar2(img):
	full = [[0 for i in range(len(img[0]))]for j in range(len(img))]
	for i in range(1,len(img)-1):	
		for j in range(1,len(img[i])-1):
			go=[]
			go.append(img[i+1][j])
			go.append(img[i-1][j])	
			go.append(img[i][j+1])
			go.append(img[i][j-1])
			set = 0
			for c in range(len(go)):
				if go.count(go[c])>=3:
					set = 1
					full[i][j] = go[c]
			if set == 0:
				full[i][j] = img[i][j]
	return full

main()
