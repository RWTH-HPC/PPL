import random


file = open("new_videofile.txt", "w")
for i in range(104*656*744):
    file.write(str(round(random.uniform(0.0, 256.0), 1)) + "\n")
file.close()
