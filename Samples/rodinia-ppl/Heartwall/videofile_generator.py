import random

with open('videofile.txt', 'w') as f:
    for _ in range(104):
        for _ in range(656):
            for _ in range(744):
                f.write("%.2f\n" % random.uniform(0, 256))
