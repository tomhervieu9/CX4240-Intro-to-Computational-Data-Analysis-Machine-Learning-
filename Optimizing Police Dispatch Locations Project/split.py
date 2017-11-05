def fun():
    f = open("houseList.csv", "r")
    M = f.readlines()
    f.close
    print(M[1])
    print(M[1].split(",")[1])
    i = 0
    f = open("Day.csv", "w")
    f.write(M[0])
    g = open("Night.csv", "w")
    g.write(M[0])
    while i < len(M):
        if i != 0:
            row = M[i]
            pri = row.split(",")[1]
            if pri == "High" or pri == "Medium":
                timeStamp = row.split(",")[0]
                hour = int(timeStamp.split(" ")[1].split(":")[0])
                AP = timeStamp.split(" ")[2]
                if (AP == "AM" and hour >= 6) or (AP == "PM" and hour < 6):
                    f.write(row)
                else:
                    g.write(row)
        i = i + 1
    f.close()
    g.close()
