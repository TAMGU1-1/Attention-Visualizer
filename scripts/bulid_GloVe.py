import json
N = 10000
out = {}
with open("C:\\Users\\ingyo\\Downloads\\glove.6B.50d.txt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i>=N:
            break
        parts = line.split()
        word = parts[0]
        vec = [round(float(x), 4) for x in parts[1:]] 
        out[word] = vec

with open("C:\\Users\\ingyo\\Downloads\\glove-50d.json", "w", encoding="utf-8") as f:
    json.dump(out,f)
print("done:", len(out), "words")