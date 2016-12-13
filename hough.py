import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import scipy.ndimage as sp
import copy
def main():
	img = cv2.imread('HoughCircles.jpg',0)
	img = sp.gaussian_filter(img, sigma=3)
	array = plot(img)
	r=peaks(array)
	print(r)
	am =atmax(array)
	print(am)
	#r=[0,125,255]
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
	forlorn = []
	for r in range(40,80):
		print(r)
		full = [[0 for i in range(len(img[0]))]for j in range(len(img))]
		toote=copy.deepcopy(toot)
		for i in range(len(toot[0])/4):
			for j in range(len(toot)/4):
				if toot[j*4][i*4]==255:
					full = softcircle(i*4,j*4,r,toote,full)
		for i in range(len(toot[0])):
			for j in range(len(toot)):
				if 170<full[j][i]:
					n = hardcircle(i,j,r,len(img[0]),len(img))
					forlorn.append(n[:])
					save(n,'gr'+str(r))
		save(full,'Circles/gr'+str(r))
	full = [[0 for i in range(len(img[0]))]for j in range(len(img))]
	for k in range(len(forlorn)):
		for i in range(len(forlorn[k][0])):
			for j in range(len(forlorn[k])):
				if forlorn[k][j][i]==255:
					full[j][i]=255
	save(toot,'gr')
	save(full,'ending')

def hardcircle(x,y,r,dx,dy):
	img = [[0 for i in range(dx)]for j in range(dy)]
	for i in range(360):
		if 0<int(x+r*math.cos(i))<len(img[0]) and 0<int(y+r*math.sin(i))<len(img):
			#print((int(y+r*math.sin(i)),int(x+r*math.cos(i))))
			img[int(y+r*math.sin(i))][int(x+r*math.cos(i))]=255
	return img

def softcircle(x,y,r,img,full):
	for i in range(360):
		if 0<int(x+r*math.cos(i))<len(img[0]) and 0<int(y+r*math.sin(i))<len(img):
			#print((int(y+r*math.sin(i)),int(x+r*math.cos(i))))
			full[int(y+r*math.sin(i))][int(x+r*math.cos(i))]+=10
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
			#waga[i][j]=count
	#print(full)
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
	#plt.show()
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
		#print(i)s
		final.append((lis[i]+lis[i+1]+lis[i-1])/3)
	final.append((lis[len(lis)-1]+lis[len(lis)-2])/2)
	return final

def save(nplist,outfile):
	nplist = np.asarray(nplist,dtype='float32')
	nplist = cv2.cvtColor(nplist,cv2.COLOR_GRAY2BGR)
	#print(wags)
	#cv2.namedWindow(outfile, cv2.WINDOW_NORMAL)
	#cv2.imshow(outfile, nplist)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
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
					#print(thresh*k)
					#print(thresh*k)
					if thresh*k not in y:
						y.append(thresh*k)
					full[i][j]=thresh*k
	y.sort()
	return full,y

def similar2(img):
	full = [[0 for i in range(len(img[0]))]for j in range(len(img))]
	for i in range(1,len(img)-1):	
		for j in range(1,len(img[i])-1):
			go=[]
			#go.append(img[i][j])
			go.append(img[i+1][j])
			go.append(img[i-1][j])	
			go.append(img[i][j+1])
			go.append(img[i][j-1])
			set = 0
			for c in range(len(go)):
				#print(go)
				if go.count(go[c])>=3:# and go[c]!=255:
					set = 1
					full[i][j] = go[c]
			if set == 0:
				full[i][j] = img[i][j]
	return full

main()
