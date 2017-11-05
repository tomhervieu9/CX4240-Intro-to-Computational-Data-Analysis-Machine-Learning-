import random


def kMeans(K, fileIn, fileOut):
    f = open(fileIn)
    N = f.readlines()
    f.close()
    print(N[1].split(",")[-2][2:])
    print (N[1].split(",")[-1][:-3])
    
    D = []
    i = 1
    minLat = 39.192463
    minLon = -76.739283
    maxLon = -76.526010
    maxLat = 39.375513
    assignment = []
    while i < len(N):
        
        lat = N[i].split(",")[-2][2:]
        lon = N[i].split(",")[-1][:-3]
        if len( lat )> 1 and len (lon) > 1:
            lat = float(lat)
            lon = float(lon)
            if lat < maxLat and lat > minLat and lon < maxLon and lon > minLon:
                D.append((float(lat),float(lon)))
                assignment.append(0)
            
        
        i = i + 1
    

    centroids = []
    for i in range(int(K)):
        randLat = (maxLat - minLat)*random.random() + minLat
        randLon = (maxLon - minLon)*random.random() + minLon
        centroids.append((randLat, randLon))
    print(centroids)

    it = 0
    centerMoved = True
    while centerMoved:
        print(it)
        it = it + 1
        centerMoved = False

        for i in range(len(D)):
            temp = assignment[i]
            minD = abs(D[i][0] - centroids[temp][0]) + abs(D[i][1] - centroids[temp][1])
            for j in range(int(K)):
                if abs(D[i][0] - centroids[j][0]) + abs(D[i][1] - centroids[j][1]) < minD:
                    minD = abs(D[i][0] - centroids[j][0]) + abs(D[i][1] - centroids[j][1])
                    assignment[i] = j
        for j in range(int(K)):
            count = 0.0
            sumL = (0,0)
            for i in range(len(D)):
                if assignment[i] == j:
                    count = count + 1
                    sumL = (sumL[0] + D[i][0], sumL[1] + D[i][1])
            if count != 0 :
                newCenter = (sumL[0]/count, sumL[1]/count)
                if newCenter != centroids[j]:
                    centerMoved = True
                    centroids[j] = newCenter
    print(centroids)
    centCount = []
    f = open(fileOut, "w")
    f.write("Latitude, Longitute, center count \n")
    for j in range(int(K)):
        count = 0
        for i in range(len(D)):
            if assignment[i] == j:
                count = count + 1
        centCount.append(count)
        if centCount != 0:
            f.write(str(centroids[j][0]) +"," + str(centroids[j][1]) + ","
                    + str(centCount[j]) + "\n")
    f.close()
        
if __name__ == '__main__': 
 kMeans(300, 'Day.csv', 'DayCenters.csv')
 kMeans(300, 'Night.csv', 'NightCenters.csv')
            
            
    
